import os
import uuid
import sqlite3
import platform
from azure.cognitiveservices.vision.face import FaceClient
from flask import Flask, render_template, request, redirect, url_for
from azure.storage.blob import BlobServiceClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from azure.communication.email import EmailClient
import config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ruta según sistema operativo
if platform.system() == 'Windows':
    DB_PATH = os.path.join(os.path.dirname(__file__), 'database', 'lostfinder.db')
else:
    DB_PATH = '/home/database/lostfinder.db'

# ──────────────────────────────────────────────
# CLIENTES DE AZURE
# ──────────────────────────────────────────────

vision_client = ComputerVisionClient(
    config.COMPUTER_VISION_ENDPOINT,
    CognitiveServicesCredentials(config.COMPUTER_VISION_KEY)
)

face_client = FaceClient(
    config.FACE_ENDPOINT,
    CognitiveServicesCredentials(config.FACE_KEY)
)

blob_service_client = BlobServiceClient.from_connection_string(
    config.BLOB_CONNECTION_STRING
)

email_client = EmailClient.from_connection_string(
    config.COMMUNICATION_CONNECTION_STRING
)

# ──────────────────────────────────────────────
# BASE DE DATOS SQLITE
# ──────────────────────────────────────────────

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS objetos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT,
            descripcion TEXT,
            etiquetas TEXT,
            confianza REAL,
            imagen_url TEXT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            lugar TEXT,
            usuario TEXT,
            personas_detectadas INTEGER,
            detalle_rostros TEXT,
            entregado_objetos_perdidos INTEGER DEFAULT 0
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alertas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            termino_busqueda TEXT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ──────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def subir_a_blob(ruta_local, nombre_archivo):
    blob_client = blob_service_client.get_blob_client(
        container=config.BLOB_CONTAINER_NAME,
        blob=nombre_archivo
    )
    with open(ruta_local, 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)

    url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{config.BLOB_CONTAINER_NAME}/{nombre_archivo}"
    return url


def analizar_imagen(ruta_local):
    with open(ruta_local, 'rb') as imagen:
        caracteristicas = [
            VisualFeatureTypes.description,
            VisualFeatureTypes.tags,
            VisualFeatureTypes.objects
        ]
        resultado = vision_client.analyze_image_in_stream(imagen, caracteristicas)

    descripcion = ""
    if resultado.description and resultado.description.captions:
        descripcion = resultado.description.captions[0].text

    etiquetas = [tag.name for tag in resultado.tags if tag.confidence > 0.7]
    nombre_objeto = etiquetas[0] if etiquetas else "Objeto desconocido"
    confianza = resultado.tags[0].confidence * 100 if resultado.tags else 0

    return {
        "nombre": nombre_objeto,
        "descripcion": descripcion,
        "etiquetas": ", ".join(etiquetas),
        "confianza": round(confianza, 2)
    }


def analizar_rostros(ruta_local):
    try:
        with open(ruta_local, 'rb') as imagen:
            rostros = face_client.face.detect_with_stream(
                imagen,
                detection_model='detection_01',
                recognition_model='recognition_04',
                return_face_attributes=['age', 'gender', 'emotion']
            )

        if not rostros:
            return {"personas_detectadas": 0, "detalle": "No se detectaron personas en la imagen"}

        detalles = []
        for rostro in rostros:
            attrs = rostro.face_attributes
            emocion_principal = max(
                attrs.emotion.__dict__.items(),
                key=lambda x: x[1] if isinstance(x[1], float) else 0
            )
            detalles.append(
                f"Persona detectada — Edad aproximada: {int(attrs.age)}, "
                f"Emoción: {emocion_principal[0]}"
            )

        return {"personas_detectadas": len(rostros), "detalle": " | ".join(detalles)}

    except Exception as e:
        print(f"Error en Face API: {e}")
        return {"personas_detectadas": 0, "detalle": "No se pudo analizar la imagen con Face API"}


def enviar_notificacion(email_destino, termino, objeto):
    """Envía email de notificación cuando se sube un objeto relacionado a una búsqueda"""
    try:
        mensaje = {
            "senderAddress": config.SENDER_EMAIL,
            "recipients": {"to": [{"address": email_destino}]},
            "content": {
                "subject": f"🔍 LostFinder AI — Encontramos un objeto relacionado a '{termino}'",
                "html": f"""
                <html>
                <body style="font-family: Arial, sans-serif; padding: 20px;">
                    <h2 style="color: #0078d4;">🔍 LostFinder AI</h2>
                    <p>Hola, registraste una alerta para búsquedas relacionadas con <strong>"{termino}"</strong>.</p>
                    <p>Se acaba de registrar un objeto que podría interesarte:</p>
                    <table style="border-collapse: collapse; width: 100%;">
                        <tr>
                            <td style="padding: 8px;"><strong>Objeto:</strong></td>
                            <td style="padding: 8px;">{objeto['nombre']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><strong>Descripción:</strong></td>
                            <td style="padding: 8px;">{objeto['descripcion']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><strong>Lugar:</strong></td>
                            <td style="padding: 8px;">{objeto['lugar']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><strong>Reportado por:</strong></td>
                            <td style="padding: 8px;">{objeto['usuario']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><strong>Entregado a objetos perdidos:</strong></td>
                            <td style="padding: 8px;">{'✅ Sí' if objeto['entregado_objetos_perdidos'] else '❌ No'}</td>
                        </tr>
                    </table>
                    <br>
                    <img src="{objeto['imagen_url']}" style="max-width: 300px; border-radius: 8px;">
                    <br><br>
                    <p style="color: #999; font-size: 0.85rem;">LostFinder AI — Powered by Microsoft Azure</p>
                </body>
                </html>
                """
            }
        }
        email_client.begin_send(mensaje)
    except Exception as e:
        print(f"Error enviando email: {e}")


def notificar_alertas(etiquetas, objeto):
    """Busca alertas registradas y notifica si hay coincidencias"""
    try:
        db = get_db()
        alertas = db.execute('SELECT * FROM alertas').fetchall()
        db.close()

        etiquetas_lista = [e.strip().lower() for e in etiquetas.split(',')]

        for alerta in alertas:
            termino = alerta['termino_busqueda'].lower()
            if any(termino in etiqueta for etiqueta in etiquetas_lista):
                enviar_notificacion(alerta['email'], alerta['termino_busqueda'], objeto)
    except Exception as e:
        print(f"Error en notificar_alertas: {e}")

# ──────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────

@app.route('/')
def index():
    db = get_db()
    objetos = db.execute('SELECT * FROM objetos ORDER BY fecha DESC LIMIT 6').fetchall()
    db.close()
    return render_template('index.html', objetos=objetos)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        archivo = request.files.get('imagen')
        lugar = request.form.get('lugar', 'No especificado')
        usuario = request.form.get('usuario', 'Anónimo')
        entregado = 1 if request.form.get('entregado') else 0

        if not archivo or not allowed_file(archivo.filename):
            return render_template('upload.html', error="Solo se permiten imágenes JPG o PNG.")

        nombre_unico = f"{uuid.uuid4().hex}_{archivo.filename}"
        ruta_temporal = os.path.join(app.config['UPLOAD_FOLDER'], nombre_unico)
        archivo.save(ruta_temporal)

        analisis = analizar_imagen(ruta_temporal)
        rostros = analizar_rostros(ruta_temporal)
        imagen_url = subir_a_blob(ruta_temporal, nombre_unico)
        os.remove(ruta_temporal)

        db = get_db()
        db.execute('''
            INSERT INTO objetos (nombre, descripcion, etiquetas, confianza, imagen_url, lugar, usuario, personas_detectadas, detalle_rostros, entregado_objetos_perdidos)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analisis['nombre'],
            analisis['descripcion'],
            analisis['etiquetas'],
            analisis['confianza'],
            imagen_url,
            lugar,
            usuario,
            rostros['personas_detectadas'],
            rostros['detalle'],
            entregado
        ))
        db.commit()
        objeto_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
        objeto = db.execute('SELECT * FROM objetos WHERE id = ?', (objeto_id,)).fetchone()
        db.close()

        notificar_alertas(analisis['etiquetas'], objeto)

        return redirect(url_for('resultado', objeto_id=objeto_id))

    return render_template('upload.html')


@app.route('/resultado/<int:objeto_id>')
def resultado(objeto_id):
    db = get_db()
    objeto = db.execute('SELECT * FROM objetos WHERE id = ?', (objeto_id,)).fetchone()
    db.close()
    if not objeto:
        return redirect(url_for('index'))
    return render_template('results.html', objeto=objeto)


@app.route('/search')
def search():
    query = request.args.get('q', '')
    email_alerta = request.args.get('email_alerta', '')

    if query and email_alerta:
        try:
            db = get_db()
            db.execute(
                'INSERT INTO alertas (email, termino_busqueda) VALUES (?, ?)',
                (email_alerta, query)
            )
            db.commit()
            db.close()
        except Exception as e:
            print(f"Error guardando alerta: {e}")

    db = get_db()
    if query:
        objetos = db.execute('''
            SELECT * FROM objetos
            WHERE nombre LIKE ? OR etiquetas LIKE ? OR descripcion LIKE ?
            ORDER BY fecha DESC
        ''', (f'%{query}%', f'%{query}%', f'%{query}%')).fetchall()
    else:
        objetos = db.execute('SELECT * FROM objetos ORDER BY fecha DESC').fetchall()
    db.close()
    return render_template('search.html', objetos=objetos, query=query, email_alerta=email_alerta)


# ──────────────────────────────────────────────
# INICIO
# ──────────────────────────────────────────────

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs('uploads', exist_ok=True)
init_db()

if __name__ == '__main__':
    app.run(debug=True)