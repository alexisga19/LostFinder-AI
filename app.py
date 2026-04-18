import os
import uuid
import sqlite3
from azure.cognitiveservices.vision.face import FaceClient
from flask import Flask, render_template, request, redirect, url_for
from azure.storage.blob import BlobServiceClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ruta persistente para App Service
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

# ──────────────────────────────────────────────
# BASE DE DATOS SQLITE
# ──────────────────────────────────────────────

def init_db():
    os.makedirs('/home/database', exist_ok=True)
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
            detalle_rostros TEXT
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
    """Sube imagen a Azure Blob Storage y retorna la URL pública"""
    blob_client = blob_service_client.get_blob_client(
        container=config.BLOB_CONTAINER_NAME,
        blob=nombre_archivo
    )
    with open(ruta_local, 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)

    url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{config.BLOB_CONTAINER_NAME}/{nombre_archivo}"
    return url


def analizar_imagen(ruta_local):
    """Analiza imagen con Azure Computer Vision y retorna resultados"""
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
    """Detecta si hay personas en la imagen con Face API"""
    try:
        with open(ruta_local, 'rb') as imagen:
            rostros = face_client.face.detect_with_stream(
                imagen,
                detection_model='detection_01',
                recognition_model='recognition_04',
                return_face_attributes=['age', 'gender', 'emotion']
            )

        if not rostros:
            return {
                "personas_detectadas": 0,
                "detalle": "No se detectaron personas en la imagen"
            }

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

        return {
            "personas_detectadas": len(rostros),
            "detalle": " | ".join(detalles)
        }

    except Exception as e:
        print(f"Error en Face API: {e}")
        return {
            "personas_detectadas": 0,
            "detalle": "No se pudo analizar la imagen con Face API"
        }

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

        if not archivo or not allowed_file(archivo.filename):
            return render_template('upload.html', error="Solo se permiten imágenes JPG o PNG.")

        # Guardar imagen temporalmente
        nombre_unico = f"{uuid.uuid4().hex}_{archivo.filename}"
        ruta_temporal = os.path.join(app.config['UPLOAD_FOLDER'], nombre_unico)
        archivo.save(ruta_temporal)

        # Analizar con Computer Vision
        analisis = analizar_imagen(ruta_temporal)

        # Analizar con Face API
        rostros = analizar_rostros(ruta_temporal)

        # Subir a Blob Storage
        imagen_url = subir_a_blob(ruta_temporal, nombre_unico)

        # Eliminar archivo temporal
        os.remove(ruta_temporal)

        # Guardar en base de datos
        db = get_db()
        db.execute('''
            INSERT INTO objetos (nombre, descripcion, etiquetas, confianza, imagen_url, lugar, usuario, personas_detectadas, detalle_rostros)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analisis['nombre'],
            analisis['descripcion'],
            analisis['etiquetas'],
            analisis['confianza'],
            imagen_url,
            lugar,
            usuario,
            rostros['personas_detectadas'],
            rostros['detalle']
        ))
        db.commit()
        objeto_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
        db.close()

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
    return render_template('search.html', objetos=objetos, query=query)


# ──────────────────────────────────────────────
# INICIO
# ──────────────────────────────────────────────

# Inicializar carpetas y base de datos siempre
os.makedirs('/home/database', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
init_db()

if __name__ == '__main__':
    app.run(debug=True)