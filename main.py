import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, List
from groq import Groq

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

db = {"categories": {}, "merchants": {}, "triggers": {}, "customers": {}}


class BotResponse(BaseModel):
    action: Literal["send", "end", "wait"] = Field(default="send")
    body: str = Field(description="WhatsApp message body.")
    cta: str = Field(description="CTA type: 'YES/STOP', 'open_ended', 'none'.")
    send_as: Literal["vera", "merchant_on_behalf"] = Field(default="vera")
    suppression_key: str = Field(default="default")
    rationale: str = Field(default="Outreach")


class TickAction(BaseModel):
    trigger_id: str
    merchant_id: str
    body: str
    cta: str = "YES/STOP"
    send_as: str = "vera"
    suppression_key: str = "tick_outreach"
    rationale: str = "Triggered engagement"


@app.get("/v1/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/metadata")
def metadata():
    return {"team_name": "Vera_Upgrade", "model": "openai/gpt-oss-20b"}


@app.post("/v1/context")
def receive_context(payload: Dict[str, Any]):
    scope = payload.get("scope")
    cid = payload.get("context_id")
    data = payload.get("payload", {})

    if scope == "category":
        db["categories"][cid] = data
    elif scope == "merchant":
        db["merchants"][cid] = data
    elif scope == "trigger":
        db["triggers"][cid] = data
    elif scope == "customer":
        db["customers"][cid] = data

    return {"status": "ok", "accepted": True}


@app.post("/v1/tick")
def process_tick(payload: Dict[str, Any]):
    available_triggers = payload.get("available_triggers", [])
    actions = []

    for tid in available_triggers:
        trig = db["triggers"].get(tid, {})
        mid = (
            trig.get("payload", {}).get("merchant_id")
            or list(db["merchants"].keys())[0]
            if db["merchants"]
            else "m_test"
        )
        merchant = db["merchants"].get(mid, {})
        cat_slug = merchant.get("category_slug", "")
        category = db["categories"].get(cat_slug, {})

        m_name = merchant.get("identity", {}).get("name", "there")
        owner = merchant.get("identity", {}).get("owner_first_name", "Clinic Director")
        locality = merchant.get("identity", {}).get("locality", "your area")
        t_kind = trig.get("kind", "update")
        active_offers = [
            o.get("title")
            for o in merchant.get("offers", [])
            if o.get("status") == "active"
        ]

        prompt = f"""
        Compose a WhatsApp message for {m_name} (Owner: Dr. {owner}, Location: {locality}).
        Category: {cat_slug} (Strict Tone: {category.get('voice', {}).get('tone', 'professional')} - NO casual greetings).
        Trigger: {t_kind}
        Details: {trig.get('payload', {})}
        Active Offers: {active_offers}
        
        Guidelines to maximize relevance:
        - Specificity: Quote exact numbers/dates from the details. You MUST include a pseudo-citation (e.g., "According to your latest {t_kind} report..." or "Based on recent regulatory updates...").
        - Merchant Fit: Greet Dr. {owner} and mention {locality}.
        - Engagement: State a clear clinical or business risk of inaction. End with: 'Reply YES to proceed'.
        
        Respond with strict JSON:
        {{"body": "message text", "cta": "YES/STOP", "rationale": "reason"}}
        """

        try:
            res = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You output strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                model="openai/gpt-oss-20b",
                response_format={"type": "json_object"},
            )
            import json

            parsed = json.loads(res.choices[0].message.content)
            body_text = parsed.get(
                "body",
                f"Hi {m_name}, you have a new update regarding {t_kind}. Reply YES to review.",
            )
            cta_text = parsed.get("cta", "YES/STOP")
            rat = parsed.get("rationale", "Proactive trigger")
        except Exception:
            body_text = f"Hi {m_name}, your listing has an update on {t_kind}. Reply YES to review."
            cta_text = "YES/STOP"
            rat = "Fallback trigger response"

        actions.append(
            {
                "trigger_id": tid,
                "merchant_id": mid,
                "body": body_text,
                "cta": cta_text,
                "send_as": "vera",
                "suppression_key": f"trig_{tid}",
                "rationale": rat,
            }
        )

    return {"actions": actions}


@app.post("/v1/reply", response_model=BotResponse)
def generate_reply(payload: Dict[str, Any]):
    user_msg = payload.get("message", "").lower()
    turn = payload.get("turn_number", 1)

    if any(w in user_msg for w in ["stop", "spam", "unsubscribe", "don't message"]):
        return BotResponse(
            action="end",
            body="Understood, we won't message you again. Best wishes!",
            cta="none",
            send_as="vera",
            suppression_key="stop",
            rationale="Merchant opted out.",
        )

    if "thank you for contacting" in user_msg or "automated" in user_msg or turn > 3:
        return BotResponse(
            action="end",
            body="Understood! I will reach out to the manager directly.",
            cta="none",
            send_as="vera",
            suppression_key="auto_reply",
            rationale="Detected canned auto-reply.",
        )

    if any(
        w in user_msg
        for w in ["lets do it", "let's do it", "proceed", "yes", "whats next"]
    ):
        return BotResponse(
            action="send",
            body="Done! I've initiated your profile update. Next, should we draft your new Google post?",
            cta="YES/STOP",
            send_as="vera",
            suppression_key="intent_act",
            rationale="Switched to action mode on explicit commitment.",
        )

    return BotResponse(
        action="send",
        body="Hi there! I noticed your listing could use a quick refresh on Google. Should we take a look?",
        cta="open_ended",
        send_as="vera",
        suppression_key="default",
        rationale="Standard greeting.",
    )
