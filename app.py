from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
import re
from datetime import datetime, timedelta
from dateutil import parser
from pyngrok import ngrok
from transformers import pipeline
from eventos import EVENTOS

# ---------- Configuración ----------
PORT = 5000
DEVICE = "cuda" if False else "cpu"
SCORE_THRESHOLD = 0.6

app = Flask(__name__)

# ---------- Modelo Zero‑Shot ----------
print("🔍 Cargando modelo de clasificación zero-shot…")
classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",
    device=0 if DEVICE == "cuda" else -1,
    hypothesis_template="This text is {}."
)
INTENT_LABELS = ["greeting", "agenda", "type", "date", "other"]

# ---------- Auxiliares ----------
SALUDOS_MANUALES = ["hola", "buenos", "buenas", "hey", "qué tal", "hello", "hi"]
TIPOS = ["música", "teatro", "feria", "danza", "concierto", "taller"]
WEEKDAYS_ES = {
    "lunes": 0, "martes": 1, "miercoles": 2, "miércoles": 2, "jueves": 3,
    "viernes": 4, "sabado": 5, "sábado": 5, "domingo": 6
}

def format_summary(e: dict) -> str:
    return f"• {e['fecha']} – {e['titulo']} ({e.get('tipo', 'general')}) en {e['lugar']}"

def events_for_date(fecha: str):
    return [e for e in EVENTOS if fecha in e['fecha']]

def events_in_range(start: datetime, end: datetime):
    out = []
    for e in EVENTOS:
        try:
            raw = e['fecha'].split()[0].replace('-', '/')
            day = parser.parse(raw, dayfirst=True)
        except Exception:
            continue
        if start.date() <= day.date() <= end.date():
            out.append(e)
    return out

def next_weekday(base: datetime, weekday: int):
    delta = (weekday - base.weekday() + 7) % 7
    delta = 7 if delta == 0 else delta
    return base + timedelta(days=delta)

# --- lógica de envío según fuentes ---

def send_events(resp: MessagingResponse, eventos: list, header: str | None = None):
    if not eventos:
        return
    fuentes = {e['fuente'] for e in eventos}
    if len(eventos) > 1 and len(fuentes) == 1:
        lines = []
        if header:
            lines.append(header)
        lines += [format_summary(ev) for ev in eventos]
        lines.append(f"Más info: {list(fuentes)[0]}")
        resp.message("\n".join(lines))
    else:
        if header:
            resp.message(header)
        for ev in eventos:
            resp.message(f"{format_summary(ev)}\nFuente: {ev['fuente']}")

# ---------- Webhook ----------
@app.route('/webhook', methods=['POST'])
def webhook():
    incoming = request.form.get('Body', '').strip()
    text = incoming.lower()
    resp = MessagingResponse()
    now = datetime.now()

    # Saludos manuales
    if any(text.startswith(g) for g in SALUDOS_MANUALES):
        resp.message(
            "¡Hola! 👋 Soy tu asistente cultural de San Miguel de Tucumán.\n"
            "📌 Ejemplos de consulta:\n"
            "• eventos hoy\n• eventos fin de semana\n• eventos música\n• eventos 10/05\n¿En qué te ayudo hoy?"
        )
        return Response(str(resp), mimetype='application/xml')

    # Clasificación
    res = classifier(text, INTENT_LABELS)
    label, score = res['labels'][0], res['scores'][0]
    if score < SCORE_THRESHOLD:
        label = "other"

    eventos = []
    encabezado = None

    # --- Intents ---
    if label == "greeting":
        resp.message("¡Hola nuevamente! ¿Qué te gustaría saber de la agenda cultural?")

    elif label == "agenda":
        if 'hoy' in text:
            eventos = events_for_date(now.strftime('%d/%m'))
            encabezado = 'Eventos para hoy:'
        elif 'fin de semana' in text:
            fri = now + timedelta((4 - now.weekday()) % 7)
            sun = fri + timedelta(days=2)
            eventos = events_in_range(fri, sun)
            encabezado = 'Eventos para este fin de semana:'
        elif 'mes' in text:
            start = now.replace(day=1)
            end = (start.replace(month=(start.month % 12) + 1, day=1) - timedelta(days=1))
            eventos = events_in_range(start, end)
            encabezado = 'Eventos para este mes:'
        else:
            m = re.search(r"(\d{1,2}[/-]\d{1,2})", text)
            if m:
                fecha_str = m.group(1).replace('-', '/')
                eventos = events_for_date(fecha_str)
                encabezado = f'Eventos para el {fecha_str}:'
        if eventos:
            send_events(resp, eventos, encabezado)
        else:
            resp.message("No tengo eventos para esa fecha, pero puedo avisarte si aparece algo 😊")

    elif label == "type":
        tipo = next((t for t in TIPOS if t in text), None)
        if tipo:
            eventos = [e for e in EVENTOS if tipo in e.get('tipo', '').lower()]
            encabezado = f'Eventos de {tipo}:'
        if eventos:
            send_events(resp, eventos, encabezado)
        else:
            resp.message(f"No encontré eventos de {tipo or 'ese tipo'}, pero te aviso si aparece alguno 😊")

    elif label == "date":
        fecha_str = None
        m = re.search(r"(\d{1,2}[/-]\d{1,2})", text)
        if m:
            fecha_str = m.group(1).replace('-', '/')
        else:
            for wd_es, idx in WEEKDAYS_ES.items():
                if wd_es in text:
                    fecha_str = next_weekday(now, idx).strftime('%d/%m')
                    break
        if fecha_str:
            eventos = events_for_date(fecha_str)
            encabezado = f'Eventos para el {fecha_str}:'
            if eventos:
                send_events(resp, eventos, encabezado)
            else:
                resp.message(f"No hay eventos para el {fecha_str}, pero puedo avisarte cuando surja algo 😊")
        else:
            resp.message("No pude reconocer la fecha. Probá con DD/MM o un día (ej. sábado).")

    else:
        resp.message("Disculpá, no entendí. Probá con 'eventos hoy' o 'eventos música'.")

    return Response(str(resp), mimetype='application/xml')

# ---------- Run ----------
if __name__ == '__main__':
    public_url = ngrok.connect(PORT)
    print(f"* ngrok tunnel: {public_url}")
    app.run(host='0.0.0.0', port=PORT)
