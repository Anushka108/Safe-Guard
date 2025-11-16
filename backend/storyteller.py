"""Robust StoryTeller with optional LLaMA backend.

This module will use llama-cpp-python if available and a model
path is provided. If the package or model is not available, it
falls back to a small deterministic explanation generator so the
API still returns useful text.
"""
import os

try:
    from llama_cpp import Llama
    _LLM_AVAILABLE = True
except Exception:
    Llama = None
    _LLM_AVAILABLE = False


class StoryTeller:
    def __init__(self, model_path: str | None = None):
        """Initialize StoryTeller.

        model_path: optional path to a gguf/ggml model used by llama_cpp.
        If llama_cpp isn't installed or model_path is missing, a simple
        rule-based fallback will be used.
        """
        self.model_path = model_path
        self.llm = None

        if _LLM_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                # Keep n_ctx modest to avoid huge memory usage
                self.llm = Llama(model_path=model_path, n_ctx=2048)
            except Exception as e:
                print(f"Warning: failed to initialize Llama model: {e}")
                self.llm = None

    def explain(self, hip: float, knee: float, shoulder: float, risk: float) -> str:
        """Return an explanation string given the final angles and risk.

        If an LLM is available, delegate to it; otherwise return a
        concise, human-readable fallback explanation with 2-3 corrections.
        """
        if self.llm is not None:
            prompt = f"""
You are a biomechanics expert. Provide a short plain-language
explanation (2-4 sentences) of the injury risk and 2 practical
corrections the user can apply.

Hip angle: {hip:.2f}°
Knee angle: {knee:.2f}°
Shoulder angle: {shoulder:.2f}°
Predicted injury risk: {risk:.2f}%
"""
            try:
                out = self.llm(prompt, max_tokens=200)
                # llama-cpp-python usually returns a dict with choices/text
                if isinstance(out, dict):
                    # Try expected structure
                    choices = out.get("choices") or out.get("generations")
                    if choices and isinstance(choices, list):
                        first = choices[0]
                        text = first.get("text") if isinstance(first, dict) else str(first)
                        return text or str(out)
                return str(out)
            except Exception as e:
                print(f"Warning: LLM generation failed: {e}")

        # Fallback deterministic explanation
        parts = []
        parts.append(f"Predicted injury risk is {risk:.1f}%.")

        # Simple heuristic-based advice
        advice = []
        if hip < 70:
            advice.append("Your hip appears very closed — try increasing hip extension and keep hips level.")
        elif hip > 120:
            advice.append("Your hip is hyperextended — engage your core and avoid locking the hip.")
        else:
            advice.append("Hip angle looks within a normal range; maintain controlled movement.")

        if knee < 80:
            advice.append("Knee is quite flexed — ensure knee tracks over toes and avoid inward collapse.")
        elif knee > 140:
            advice.append("Knee is overly extended — avoid locking and use soft bend to absorb load.")
        else:
            advice.append("Knee positioning is reasonable; keep alignment with the hip and ankle.")

        if shoulder < 30:
            advice.append("Shoulders are low — open the chest and retract the shoulder blades slightly.")
        elif shoulder > 100:
            advice.append("Shoulders are excessively raised — relax the neck and lower shoulders to reduce strain.")

        # Pick up to 3 unique advice points
        unique = []
        for a in advice:
            if a not in unique:
                unique.append(a)
            if len(unique) >= 3:
                break

        correction_lines = " ".join(unique)
        return " ".join(parts) + " " + correction_lines
