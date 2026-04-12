# Meeting Scenarios (Cartridges)

Test-Szenarien fuer die Meeting Intelligence Smoke Tests.
Jede YAML-Datei ist ein komplettes Meeting das durchgespielt werden kann.

## Format

```yaml
meta:
  title: "Szenario Name"
  difficulty: easy|medium|hard    # Wie schwer fuer den Agent
  duration_minutes: 10            # Simulierte Meeting-Laenge
  language: de                    # Hauptsprache
  tags: [standup, api, trigger]   # Fuer Filterung

participants:
  - name: "Davi"
  - name: "Max"
  - name: "Lisa"

# Hartcodierte Segmente — deterministisch, kein LLM
segments:
  - t: 0                         # Timestamp in Sekunden
    who: "Davi"
    says: "Ok, lasst uns anfangen."

  - t: 15
    who: "Max"
    says: "Meine Sorge ist..."

# Erwartungen — was MUSS passieren
expect:
  - type: trigger                # Watcher muss triggern
    after_segment: 5             # Nach dem 5. Segment

  - type: quick_ack              # Quick-Ack muss kommen
    after_segment: 5
    max_latency_ms: 5000

  - type: state_updated          # Shared State muss aktualisiert sein
    field: context_summary
    contains: "API"

  - type: action                 # Eine bestimmte Action muss kommen
    action_type: chat
    contains: "Breaking Changes"
```

## Vorhandene Szenarien

| Datei | Schwierigkeit | Testet |
|-------|--------------|--------|
| quick_standup.yaml | easy | Basics: Trigger, Quick-Ack |
| api_migration.yaml | medium | Deep Agent: State, Suche, Antwort |
| stress_triggers.yaml | hard | 5 Trigger hintereinander, Cooldown |
| no_trigger.yaml | easy | Kein Trigger — Agent soll still sein |
| long_meeting.yaml | hard | 30 Min, Transcript Summarization |
