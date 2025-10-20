# Estructura basica para una API con Python lista para deploy en Render

Encontraras todos los archivos necesarios para la creación de una API la cual usaremos Python como lógica del proyecto y usaremos HTML para poder mostrar estos datos, ya desa consumirlos o leerlos y mostrarlos.

## Archivos Importantes

- **Procfile:** Se usa principalmente para que Render pueda desplegarlo, es decir es su entorno de despliegue.

- **requirements.txt :** Archivo necesario para que el lugar donde se va a desplegar identifique que python usara estas bibliotecas y pueda descargarlas para su correcto funcionamiento (Todas las bibliotecas que uses, las deberás colocar aqui).

- **app.py :** Aqui se encontrara nuestra Api, es decir aqui colocaras la estructura de tu API, las rutas y que es lo que hara esa ruta.

- **.gitignore :** En este archivo iran aquellas rutas que no se subiran a git, lo que pueden ser entornos de desarrollo, credenciales o archivos .venv, etc

- **/templates :** En esta ruta ira tu index.html que es donde tentras tu aplicación web la cual vera el usuario

- **/static :** Se usara para alojar tu logica en JS en caso de necesitarlo.

Asi es como tendras un entorno de desarrollo para poder crear una API y este lista para el despliegue.

Para poder correrla existen dos maneras:

- Servidor local: Para poder ejecutar un servidor donde solo tu tengas acceso se usa el comando:
```bash
gunicorn --reload [nombre_del_archivo]:[variable_de_la_app]
```

- Servidor local en la red: POdran tener acceso los dispositivos conectados a la misma red que tu (LAN), se usará el comando:
```bash
gunicorn --reload --bind 0.0.0.0:8000 [nombre_del_archivo]:[variable_de_la_app]
```

### Para deploy

Para poder deployar una API de este tipo necesiatras ingresar el comnado 

```bash
web app:gunicorn
```