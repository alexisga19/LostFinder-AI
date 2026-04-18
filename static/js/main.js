// Vista previa de imagen antes de subir
document.addEventListener('DOMContentLoaded', function () {
    const inputImagen = document.getElementById('input-imagen');

    if (inputImagen) {
        inputImagen.addEventListener('change', function () {
            const archivo = this.files[0];
            if (!archivo) return;

            // Crear o reutilizar preview
            let preview = document.getElementById('preview-img');
            if (!preview) {
                preview = document.createElement('img');
                preview.id = 'preview-img';
                preview.style.cssText = 'width:100%; max-height:300px; object-fit:contain; margin-top:1rem; border-radius:8px; border:1px solid #ddd;';
                inputImagen.parentElement.appendChild(preview);
            }

            const reader = new FileReader();
            reader.onload = function (e) {
                preview.src = e.target.result;
            };
            reader.readAsDataURL(archivo);
        });
    }
});

// Esta función debe estar FUERA del DOMContentLoaded
// para que los botones del HTML puedan llamarla
function activarOpcion(tipo) {
    const input = document.getElementById('input-imagen');

    if (tipo === 'camara') {
        input.setAttribute('capture', 'environment');
    } else {
        input.removeAttribute('capture');
    }

    input.click();
}