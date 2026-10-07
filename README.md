# Reconocimiento_Facial_Y_de_Manos
# Sistema de reconocimiento de caras y poses

## 1. Descripción del proyecto

El objetivo de este proyecto es crear una aplicación capaz de utilizar una cámara para reconocer diferentes expresiones faciales, gestos de las manos y, posteriormente, otras poses corporales.

Dependiendo de lo que detecte la cámara, el programa mostrará una imagen asociada a ese gesto o expresión.

Por ejemplo:

| Detección | Identificador | Imagen mostrada |
|---|---|---|
| Cara feliz | `feliz` | Gato feliz |
| Cara triste | `triste` | Gato triste |
| Mano levantada | `mano_arriba` | Imagen de una mano levantada |
| Pulgar arriba | `pulgar_arriba` | Imagen asociada |
| Signo de paz | `paz` | Imagen asociada |

La idea es que el sistema funcione en tiempo real mientras el usuario está delante de la cámara.

---

## 2. Objetivo final

La aplicación tendrá aproximadamente esta distribución:

```text
┌─────────────────────────────────────────────────────────┐
│                    RECONOCIMIENTO                       │
├──────────────────────────────┬──────────────────────────┤
│                              │                          │
│                              │                          │
│          CÁMARA              │       IMAGEN             │
│                              │                          │
│        ┌─────────┐            │       🐱                 │
│        │ Usuario │            │     Gato feliz           │
│        └─────────┘            │                          │
│                              │                          │
│                              │   ID: feliz              │
│                              │                          │
└──────────────────────────────┴──────────────────────────┘
```

La cámara analiza continuamente al usuario.

Cuando se detecta una característica determinada:

```text
Cámara
   ↓
Detección
   ↓
Clasificación
   ↓
Identificador
   ↓
Búsqueda de imagen
   ↓
Mostrar imagen
```

---

## 3. Tecnologías iniciales

El proyecto utilizará inicialmente:

### Python

Será el lenguaje principal del proyecto.

Ventajas:

- Fácil de desarrollar y modificar.
- Gran cantidad de librerías de visión artificial.
- Buena integración con cámaras.
- Adecuado para prototipos de inteligencia artificial.

### OpenCV

Se utilizará para:

- Acceder a la cámara.
- Capturar los frames.
- Procesar imágenes.
- Crear inicialmente la ventana de visualización.

### MediaPipe

Se utilizará para detectar elementos de la persona.

Inicialmente nos interesa especialmente:

- Cara.
- Puntos faciales.
- Manos.
- Dedos.
- Posición de las manos.

Más adelante se puede incorporar detección de postura corporal.

---

# 4. Arquitectura inicial

El programa estará dividido conceptualmente en varias partes.

```text
                 ┌─────────────┐
                 │   Cámara    │
                 └──────┬──────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Captura de vídeo │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    Detección     │
              │                  │
              │ Cara / Manos     │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   Clasificador   │
              │                  │
              │ ¿Qué está        │
              │ haciendo?        │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   Identificador  │
              │                  │
              │ "feliz"          │
              │ "paz"            │
              │ "mano_arriba"    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Gestor de        │
              │ imágenes         │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Imagen asociada  │
              └──────────────────┘
```

---

# 5. Concepto de identificadores

Cada gesto o expresión tendrá un identificador único.

Ejemplo:

```text
feliz
triste
mano_arriba
mano_abajo
pulgar_arriba
pulgar_abajo
paz
```

Estos identificadores serán importantes porque separarán la detección de la presentación visual.

Por ejemplo:

```text
MediaPipe
    ↓
detecta una expresión
    ↓
clasificador
    ↓
"feliz"
    ↓
ID = feliz
    ↓
buscar imagen asociada
    ↓
gato_feliz.jpg
```

Esto permitirá cambiar las imágenes sin tener que modificar el sistema de reconocimiento.

---

# 6. Estructura inicial del proyecto

La estructura prevista será:

```text
reconocimiento-caras/
│
├── README.md
│
├── requirements.txt
│
├── main.py
│
├── camera/
│   └── camera.py
│
├── detection/
│   ├── face.py
│   └── hands.py
│
├── recognition/
│   └── classifier.py
│
├── images/
│   ├── feliz/
│   │   └── gato_feliz.jpg
│   │
│   ├── triste/
│   │   └── gato_triste.jpg
│   │
│   ├── mano_arriba/
│   │   └── mano_arriba.jpg
│   │
│   ├── pulgar_arriba/
│   │   └── pulgar_arriba.jpg
│   │
│   └── paz/
│       └── paz.jpg
│
└── config/
    └── poses.json
```

La estructura puede cambiar durante el desarrollo.

---

# 7. Fases del proyecto

No se intentará desarrollar todo el sistema de una vez.

Se seguirá una progresión para poder probar cada parte individualmente.

## Fase 1 — Cámara

Objetivo:

Conseguir abrir la cámara y mostrar el vídeo en tiempo real.

Resultado esperado:

```text
┌───────────────────────┐
│                       │
│       VÍDEO           │
│       CÁMARA          │
│                       │
└───────────────────────┘
```

Criterio para pasar de fase:

- La cámara funciona.
- El vídeo se actualiza correctamente.
- Podemos cerrar la aplicación correctamente.

---

## Fase 2 — Detección de cara

Añadir MediaPipe para detectar la cara.

Resultado esperado:

```text
┌───────────────────────┐
│                       │
│      ┌────────┐       │
│      │  CARA  │       │
│      └────────┘       │
│                       │
└───────────────────────┘
```

Además, podremos visualizar los puntos de referencia de la cara.

---

## Fase 3 — Detección de manos

Añadir detección de manos.

El programa deberá poder determinar:

- Si hay una mano.
- Si hay dos manos.
- Posición de la mano.
- Posición de los dedos.

Posteriormente utilizaremos estos datos para reconocer gestos.

---

## Fase 4 — Reconocimiento de gestos básicos

Crear las primeras reglas.

Ejemplo:

```text
Pulgar levantado
       ↓
pulgar_arriba
```

```text
Dos dedos levantados
       ↓
paz
```

```text
Mano por encima de la cabeza
       ↓
mano_arriba
```

En esta fase no necesitamos todavía inteligencia artificial compleja.

Podemos comenzar utilizando reglas basadas en los puntos detectados.

---

## Fase 5 — Reconocimiento de expresiones

Añadir reconocimiento de expresiones faciales.

Primeras categorías previstas:

```text
feliz
triste
sorprendido
enfadado
neutral
```

Esta parte probablemente requerirá un enfoque diferente al de los gestos de las manos.

---

## Fase 6 — Sistema de identificadores

Crear un sistema centralizado para las categorías.

Ejemplo:

```json
{
    "feliz": {
        "image": "images/feliz/gato_feliz.jpg"
    },
    "triste": {
        "image": "images/triste/gato_triste.jpg"
    },
    "mano_arriba": {
        "image": "images/mano_arriba/mano_arriba.jpg"
    }
}
```

Esto permitirá añadir nuevas categorías sin modificar demasiado el código.

---

## Fase 7 — Interfaz

Crear la interfaz definitiva.

La ventana tendrá dos zonas:

```text
┌──────────────────────────────────────────────────┐
│                                                  │
│       CÁMARA                │     RESULTADO      │
│                            │                    │
│       Usuario              │     Imagen         │
│                            │                    │
│                            │     ID: feliz      │
│                            │                    │
└──────────────────────────────────────────────────┘
```

---

## Fase 8 — Optimización

Una vez que todo funcione:

- Mejorar la velocidad.
- Reducir falsos positivos.
- Evitar cambios constantes de imagen.
- Añadir un sistema de confianza.
- Mejorar la interfaz.
- Añadir configuración.
- Permitir más imágenes por identificador.

---

# 8. Sistema de confianza

Uno de los problemas que tendremos que solucionar será evitar que el programa cambie de estado constantemente.

Por ejemplo:

```text
Frame 1 → feliz
Frame 2 → feliz
Frame 3 → neutral
Frame 4 → feliz
Frame 5 → neutral
```

No queremos que la imagen cambie continuamente.

Podemos implementar posteriormente un sistema de confianza:

```text
Detección:
feliz = 92%

neutral = 5%

triste = 3%
```

Y establecer un mínimo:

```text
Si confianza > 80%
    aceptar detección
```

También podremos exigir que una detección se mantenga durante varios frames.

---

# 9. Diferencia entre detección y reconocimiento

Es importante mantener estos conceptos separados.

### Detección

Responde a:

> ¿Dónde están las manos y la cara?

Por ejemplo:

```text
Cara encontrada
Mano izquierda encontrada
Mano derecha encontrada
```

### Reconocimiento

Responde a:

> ¿Qué está haciendo la persona?

Por ejemplo:

```text
feliz
mano_arriba
pulgar_arriba
paz
```

La detección proporciona los datos necesarios para que posteriormente podamos realizar el reconocimiento.

---

# 10. Primer objetivo técnico

El primer objetivo del proyecto será muy sencillo:

> Abrir la cámara del ordenador y mostrar el vídeo en tiempo real utilizando Python y OpenCV.

Todavía NO intentaremos reconocer caras, emociones ni manos.

El orden será:

```text
1. Python funcionando
       ↓
2. OpenCV instalado
       ↓
3. Cámara funcionando
       ↓
4. Vídeo en tiempo real
       ↓
5. MediaPipe
       ↓
6. Cara
       ↓
7. Manos
       ↓
8. Gestos
       ↓
9. Expresiones
       ↓
10. Identificadores
       ↓
11. Imágenes
       ↓
12. Interfaz final
```

---

# 11. Estado actual

```text
[ ] Crear proyecto Python
[ ] Crear entorno virtual
[ ] Instalar dependencias
[ ] Probar OpenCV
[ ] Abrir cámara
[ ] Mostrar vídeo
[ ] Instalar MediaPipe
[ ] Detectar cara
[ ] Detectar manos
[ ] Reconocer gestos
[ ] Reconocer expresiones
[ ] Crear identificadores
[ ] Asociar imágenes
[ ] Crear interfaz
[ ] Optimizar
```

---

# 12. Principio de desarrollo

El proyecto se desarrollará de forma incremental.

No se debe intentar implementar todas las funcionalidades simultáneamente.

Cada fase debe:

1. Implementarse.
2. Probarse.
3. Verificarse.
4. Documentarse.
5. Integrarse con la siguiente fase.

De esta forma, si aparece un problema será mucho más sencillo identificar en qué parte del sistema se encuentra.

---

# 13. Próximo paso

El siguiente paso es crear el proyecto Python desde cero.

En la primera sesión solamente haremos:

```text
Proyecto
   ↓
Entorno virtual
   ↓
OpenCV
   ↓
main.py
   ↓
Cámara
   ↓
Vídeo en tiempo real
```

Cuando esto funcione correctamente, pasaremos a la detección de la cara.

---

# 14. Salida para Microsoft Teams

Como mejora futura, se plantea permitir que el resultado del reconocimiento pueda utilizarse como cámara virtual en Microsoft Teams.

La cámara del usuario se utilizará internamente para detectar expresiones faciales y gestos, pero la salida enviada a Teams NO mostrará la cara del usuario.

El flujo previsto será:

```text
Cámara
   ↓
Reconocimiento facial y de manos
   ↓
Identificador
   ↓
Gestor de imágenes
   ↓
Solo imagen asociada
   ↓
Cámara virtual
   ↓
Microsoft Teams
```

Por ejemplo:

```text
Sonrisa
   ↓
feliz
   ↓
Gato_Feliz.jpg
   ↓
Teams muestra solamente la imagen
```

La salida para Teams deberá poder mostrar únicamente la imagen asociada al identificador detectado, sin mostrar la imagen de la cámara ni la cara del usuario.

Como posible solución técnica se estudiará el uso de OBS Studio y OBS Virtual Camera como puente entre la aplicación Python y Microsoft Teams.

Esta funcionalidad queda pendiente y no forma parte de la implementación actual.

