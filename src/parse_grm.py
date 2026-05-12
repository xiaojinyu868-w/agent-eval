def parse_grm_output(content: str) -> dict:
    """
    Parse GRM-style output: <think> + <verdict> + <score>
    Returns: {"reasoning": ..., "verdict": ..., "score": ..., "evidence": ..., "evidence_turn": ...}
    """
    import re
    # Extract <think> reasoning
    thinking = re.search(r'<think>(.+?)</think>', content, re.DOTALL)
    reasoning = thinking.group(1).strip() if thinking else ""
    
    # Extract <verdict>
    verdict_match = re.search(r'<verdict>(YES|PARTIAL|NO|N/A)</verdict>', content, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else "NO"
    
    # Extract <score>
    score_match = re.search(r'<score>(1\.0|0\.5|0\.0|N/A)</score>', content, re.IGNORECASE)
    score = score_match.group(1) if score_match else "0.0"
    
    # Extract evidence (if present)
    evidence_match = re.search(r'<evidence>([^<]+)</evidence>', content, re.IGNORECASE)
    evidence = evidence_match.group(1).strip() if evidence_match else ""
    
    evidence_turn_match = re.search(r'<evidence_turn>(\d+)</evidence_turn>', content, re.IGNORECASE)
    evidence_turn = int(evidence_turn_match.group(1)) if evidence_turn_match else None
    
    return {
        "reasoning": reasoning,
        "verdict": verdict,
        "score": score,
        "evidence": evidence,
        "evidence_turn": evidence_turn,
    }
