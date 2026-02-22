import re
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass
class PIIMatch:
    entity_type: str
    value: str
    start: int
    end: int
    confidence: str  # "high" | "medium" | "low"
class RuleBasedPIIDetector:
    def __init__(self, rules_path: str):
        self.patterns = self._load_rules(rules_path)

    def _load_rules(self, path: str) -> Dict[str, List[re.Pattern]]:
        with open(path) as f:
            config = yaml.safe_load(f)
        
        return {
            entity: [re.compile(p, re.IGNORECASE) for p in patterns]
            for entity, patterns in config["patterns"].items()
        }
    def detect(self, text: str) -> List[PIIMatch]:
        matches = []
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append(PIIMatch(
                        entity_type=entity_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=self._score_confidence(entity_type, match.group())
                    ))
        return self._deduplicate(matches)
    def _score_confidence(self, entity_type: str, value: str) -> str:
        # Tighter patterns = higher confidence
        high_confidence = {"SSN", "EMAIL", "CREDIT_CARD", "IP_ADDRESS"}
        return "high" if entity_type in high_confidence else "medium"
    def _deduplicate(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        # Remove overlapping matches, keep the more specific one
        matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
        result = []
        last_end = -1
        for match in matches:
            if match.start >= last_end:
                result.append(match)
                last_end = match.end
        return result
    def redact(self, text: str) -> str:
        matches = self.detect(text)
        # Replace from end to start so indices don't shift
        for match in sorted(matches, key=lambda m: m.start, reverse=True):
            text = text[:match.start] + f"[{match.entity_type}]" + text[match.end:]
        return text
'''
detector = RuleBasedPIIDetector("rules.yaml")
text = """
Patient John Doe, SSN 123-45-6789, DOB 03/15/1990.
Contact: john@example.com or (555) 123-4567.
Card: 4111-1111-1111-1111, IP: 192.168.1.1
"""
# Get all matches with positions
matches = detector.detect(text)
for m in matches:
    print(f"{m.entity_type}: '{m.value}' at [{m.start}:{m.end}]")

# Or just redact everything
print(detector.redact(text))
'''