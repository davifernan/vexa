/**
 * Deepgram WebSocket streaming client for the Vexa Bot.
 *
 * Replaces the batch Whisper pipeline (buffer → HTTP POST → wait)
 * with a streaming WebSocket connection (audio in → text out, ~200ms).
 *
 * Audio chunks are sent directly to Deepgram as they arrive from
 * the per-speaker audio capture. No buffering, no idle timer.
 *
 * Enabled via STT_MODE=deepgram environment variable.
 */

import WebSocket from 'ws';
import { log } from '../utils';

export interface DeepgramConfig {
  apiKey: string;
  model?: string;       // default: nova-3
  language?: string;     // default: de (or "multi" for auto-detect)
  sampleRate?: number;   // default: 16000
}

export interface DeepgramSegment {
  text: string;
  speaker: string;
  speakerId: string;
  isFinal: boolean;
  start: number;
  end: number;
  confidence: number;
  words?: Array<{ word: string; start: number; end: number; confidence: number }>;
}

type SegmentCallback = (segment: DeepgramSegment) => void;

/**
 * Per-speaker Deepgram WebSocket client.
 * One instance per speaker track — audio goes in, transcript segments come out.
 */
export class DeepgramStreamClient {
  private ws: WebSocket | null = null;
  private config: DeepgramConfig;
  private speakerId: string;
  private speakerName: string;
  private onSegment: SegmentCallback | null = null;
  private connected: boolean = false;
  private keepAliveTimer: ReturnType<typeof setInterval> | null = null;

  constructor(speakerId: string, speakerName: string, config: DeepgramConfig) {
    this.speakerId = speakerId;
    this.speakerName = speakerName;
    this.config = config;
  }

  setOnSegment(callback: SegmentCallback): void {
    this.onSegment = callback;
  }

  async connect(): Promise<void> {
    const model = this.config.model || 'nova-3';
    const language = this.config.language || 'de';
    const sampleRate = this.config.sampleRate || 16000;

    const params = new URLSearchParams({
      model,
      language,
      interim_results: 'true',
      utterance_end_ms: '1500',
      smart_format: 'true',
      encoding: 'linear16',
      sample_rate: String(sampleRate),
      channels: '1',
      punctuate: 'true',
      // No diarization needed — we already have per-speaker tracks
    });

    const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url, {
        headers: { Authorization: `Token ${this.config.apiKey}` },
      });

      this.ws.on('open', () => {
        this.connected = true;
        log(`[Deepgram] Connected for speaker "${this.speakerName}" (${model}, ${language})`);

        // Keep-alive every 10s
        this.keepAliveTimer = setInterval(() => {
          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'KeepAlive' }));
          }
        }, 10000);

        resolve();
      });

      this.ws.on('message', (data: WebSocket.Data) => {
        try {
          const msg = JSON.parse(data.toString());
          if (msg.type === 'Results') {
            this.handleResults(msg);
          }
        } catch (e) {
          // Ignore non-JSON messages
        }
      });

      this.ws.on('error', (err) => {
        log(`[Deepgram] Error for "${this.speakerName}": ${err.message}`);
        if (!this.connected) reject(err);
      });

      this.ws.on('close', (code, reason) => {
        this.connected = false;
        log(`[Deepgram] Closed for "${this.speakerName}": ${code} ${reason}`);
      });
    });
  }

  /**
   * Send audio chunk to Deepgram. Call this directly from the audio capture callback.
   * No buffering needed — Deepgram handles streaming natively.
   */
  sendAudio(audioData: Float32Array): void {
    if (!this.ws || !this.connected) return;

    // Convert Float32Array to Int16 PCM (what Deepgram expects for linear16)
    const pcm = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      const s = Math.max(-1, Math.min(1, audioData[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    try {
      this.ws.send(Buffer.from(pcm.buffer));
    } catch (e) {
      // Connection may have closed
    }
  }

  async disconnect(): Promise<void> {
    if (this.keepAliveTimer) {
      clearInterval(this.keepAliveTimer);
      this.keepAliveTimer = null;
    }

    if (this.ws && this.connected) {
      try {
        this.ws.send(JSON.stringify({ type: 'CloseStream' }));
      } catch (e) {
        // Ignore
      }
      this.ws.close();
    }
    this.ws = null;
    this.connected = false;
    log(`[Deepgram] Disconnected for "${this.speakerName}"`);
  }

  private handleResults(msg: any): void {
    const channel = msg.channel || {};
    const alternatives = channel.alternatives || [];
    if (!alternatives.length) return;

    const alt = alternatives[0];
    const text = (alt.transcript || '').trim();
    if (!text) return;

    const isFinal = msg.is_final || false;
    const start = msg.start || 0;
    const duration = msg.duration || 0;

    const words = (alt.words || []).map((w: any) => ({
      word: w.word,
      start: w.start,
      end: w.end,
      confidence: w.confidence || 0.9,
    }));

    const segment: DeepgramSegment = {
      text,
      speaker: this.speakerName,
      speakerId: this.speakerId,
      isFinal,
      start,
      end: start + duration,
      confidence: alt.confidence || 0.9,
      words,
    };

    if (this.onSegment) {
      this.onSegment(segment);
    }

    if (isFinal) {
      log(`[Deepgram] [📝 FINAL] ${this.speakerName}: "${text}"`);
    }
  }
}
