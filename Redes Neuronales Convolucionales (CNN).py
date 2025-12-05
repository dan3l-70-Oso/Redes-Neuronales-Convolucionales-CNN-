""" APLICACION PRÁCTICA DE REDES NEURONALES CONVOLUCIONES (CNN)"""

"""
CNN PARA MNIST - INSTALACIÓN AUTOMÁTICA Y EJECUCIÓN
"""

import sys
import subprocess
import importlib.util
import os

# ============================================================================
# 1. FUNCIÓN PARA VERIFICAR E INSTALAR DEPENDENCIAS                         ||
# ============================================================================

def verificar_e_instalar_dependencias():
    """Verifica e instala automáticamente todas las dependencias necesarias."""
    
    print("="*70)
    print("INSTALADOR AUTOMÁTICO DE DEPENDENCIAS")
    print("="*70)
    
    # Lista de dependencias necesarias
    dependencias = [
        'numpy>=1.19.0',
        'matplotlib>=3.3.0', 
        'scipy>=1.5.0',
        'tensorflow>=2.4.0'
    ]
    
    print("\n1. Verificando dependencias...")
    
    # Verificar si pip está disponible
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        print("pip está disponible")
    except:
        print("pip no está disponible. Instalando pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        except:
            print("No se pudo instalar pip. Por favor instálelo manualmente.")
            return False
    
    # Verificar cada dependencia
    paquetes_faltantes = []
    for dep in dependencias:
        nombre_paquete = dep.split('>=')[0] if '>=' in dep else dep.split('==')[0]
        
        # Verificar si el paquete está instalado
        spec = importlib.util.find_spec(nombre_paquete)
        if spec is None:
            print(f"{nombre_paquete} no está instalado")
            paquetes_faltantes.append(dep)
        else:
            print(f"{nombre_paquete} ya está instalado")
    
    # Instalar paquetes faltantes
    if paquetes_faltantes:
        print(f"\n2. Instalando {len(paquetes_faltantes)} dependencias faltantes...")
        print("   Esto puede tomar varios minutos. Por favor sea paciente.\n")
        
        try:
            # Actualizar pip primero
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Instalar todas las dependencias
            for dep in paquetes_faltantes:
                print(f"   Instalando {dep}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                    print(f" {dep} instalado correctamente")
                except subprocess.CalledProcessError:
                    print(f"  No se pudo instalar {dep}, intentando continuar...")
            
            print("\nInstalación completada!")
            
            # Verificar nuevamente que todo esté instalado
            print("\n3. Verificación final...")
            todo_ok = True
            for dep in paquetes_faltantes:
                nombre_paquete = dep.split('>=')[0]
                spec = importlib.util.find_spec(nombre_paquete)
                if spec is None:
                    print(f"   {nombre_paquete} aún no está disponible")
                    todo_ok = False
                else:
                    print(f"   {nombre_paquete} disponible")
            
            return todo_ok
            
        except Exception as e:
            print(f"\nError durante la instalación: {e}")
            print("\nPor favor, instale manualmente con:")
            print(f"pip install {' '.join(paquetes_faltantes)}")
            return False
    else:
        print("\nTodas las dependencias ya están instaladas!")
        return True

# ============================================================================
# 2. EJECUTAR LA INSTALACIÓN                                                ||
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("APLICACIÓN CNN PARA CLASIFICACIÓN DE DÍGITOS MNIST")
    print("="*70)
    print("\nEste script instalará automáticamente todas las dependencias necesarias.")
    print("Las dependencias requeridas son:")
    print("  • NumPy - Para operaciones matriciales")
    print("  • Matplotlib - Para visualizaciones")
    print("  • SciPy - Para operaciones de señal")
    print("  • TensorFlow - Para la red neuronal")
    
    respuesta = input("\n¿Desea continuar con la instalación? (s/n): ").lower()
    
    if respuesta != 's':
        print("\n Instalación cancelada.")
        sys.exit(0)
    
    # Ejecutar la verificación e instalación
    if verificar_e_instalar_dependencias():
        print("\n" + "="*70)
        print("INICIANDO LA APLICACIÓN CNN...")
        print("="*70)
        
        # Ahora importamos las dependencias ya instaladas
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.signal import correlate2d
            import tensorflow as tf
            
            print("Todas las bibliotecas importadas correctamente!")
            print(f"   NumPy versión: {np.__version__}")
            print(f"   TensorFlow versión: {tf.__version__}")
            
            # Configurar semillas para reproducibilidad
            np.random.seed(42)
            tf.random.set_seed(42)
            
        except ImportError as e:
            print(f"\nError al importar: {e}")
            print("   Por favor, reinicie el script.")
            sys.exit(1)
        
        # ====================================================================
        # 3. CÓDIGO DE LA APLICACIÓN CNN (SIMPLIFICADO PARA DEMOSTRACIÓN)   ||
        # ====================================================================
        
        def demostrar_convolucion():
            """Demostración simple de convolución."""
            print("\n" + "="*60)
            print("DEMOSTRACIÓN DE CONVOLUCIÓN")
            print("="*60)
            
            # Crear una imagen simple
            imagen = np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ])
            
            # Crear un filtro simple
            filtro = np.array([
                [1, 0, -2],
                [1, 0, -1],
                [1, 0, -3]
            ])
            
            print("\nImagen de 3x3:")
            print(imagen)
            
            print("\nFiltro de 3x3 (detector de bordes verticales):")
            print(filtro)
            
            # Aplicar convolución manual
            resultado = np.zeros((1, 1))
            for i in range(1):
                for j in range(1):
                    region = imagen[i:i+3, j:j+3]
                    resultado[i, j] = np.sum(region * filtro)
            
            print("\nResultado de convolución:")
            print(resultado)
            
            # Visualización
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            
            axes[0].imshow(imagen, cmap='gray', vmin=0, vmax=9)
            axes[0].set_title('Imagen original')
            axes[0].axis('off')
            
            axes[1].imshow(filtro, cmap='gray', vmin=-1, vmax=1)
            axes[1].set_title('Filtro')
            axes[1].axis('off')
            
            axes[2].imshow(resultado, cmap='gray')
            axes[2].set_title('Resultado')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        def entrenar_cnn_simple():
            """Entrena una CNN simple en MNIST."""
            print("\n" + "="*60)
            print("ENTRENANDO CNN EN MNIST")
            print("="*60)
            
            try:
                # Cargar datos MNIST
                print("\n1. Cargando dataset MNIST...")
                mnist = tf.keras.datasets.mnist
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                
                # Normalizar
                x_train, x_test = x_train / 255.0, x_test / 255.0
                
                # Añadir dimensión del canal
                x_train = x_train[..., tf.newaxis]
                x_test = x_test[..., tf.newaxis]
                
                print(f"   {len(x_train)} imágenes de entrenamiento")
                print(f"   {len(x_test)} imágenes de prueba")
                
                # Crear modelo simple
                print("\n2. Creando modelo CNN...")
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
                
                model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
                
                print("\n3. Resumen del modelo:")
                model.summary()
                
                # Entrenar por 2 épocas (rápido)
                print("\n4. Entrenando (2 épocas)...")
                history = model.fit(x_train, y_train, 
                                  epochs=2, 
                                  validation_split=0.1,
                                  verbose=1)
                
                # Evaluar
                print("\n5. Evaluando...")
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                print(f"   Precisión en prueba: {test_acc:.4f} ({test_acc*100:.1f}%)")
                
                # Visualizar algunas predicciones
                print("\n6. Visualizando predicciones...")
                predictions = model.predict(x_test[:12])
                
                fig, axes = plt.subplots(3, 4, figsize=(12, 9))
                axes = axes.flatten()
                
                for i in range(12):
                    ax = axes[i]
                    ax.imshow(x_test[i].squeeze(), cmap='gray')
                    pred_label = np.argmax(predictions[i])
                    true_label = y_test[i]
                    color = 'green' if pred_label == true_label else 'red'
                    ax.set_title(f'Real: {true_label}\nPred: {pred_label}', color=color)
                    ax.axis('off')
                
                plt.tight_layout()
                plt.show()
                
                return model, test_acc
                
            except Exception as e:
                print(f" Error: {e}")
                return None, 0
        
        def mostrar_menu():
            """Menú principal."""
            while True:
                print("\n" + "="*60)
                print("MENÚ PRINCIPAL - CNN MNIST")
                print("="*60)
                print("\n1. Demostrar operación de convolución")
                print("2. Entrenar CNN completa en MNIST")
                print("3. Salir")
                
                try:
                    opcion = int(input("\nSeleccione opción (1-3): "))
                    
                    if opcion == 1:
                        demostrar_convolucion()
                    elif opcion == 2:
                        entrenar_cnn_simple()
                    elif opcion == 3:
                        print("\n¡Gracias por usar la aplicación!")
                        break
                    else:
                        print(" Opción inválida")
                except ValueError:
                    print(" Por favor ingrese un número")
                except KeyboardInterrupt:
                    print("\n\nAplicación interrumpida")
                    break
        
        # Ejecutar el menú
        mostrar_menu()
        
    else:
        print("\n No se pudieron instalar todas las dependencias.")
        print("   Por favor, intente instalarlas manualmente:")
        print("   pip install numpy matplotlib scipy tensorflow")
        
    input("\nPresione Enter para salir...")

"""APLICACIÓN PRÁCTICA DE CNN PARA CLASIFICACIÓN DE DÍGITOS
Lo que implementa convolución manual, max pooling y CNN completa para MNIST"""

import sys
import subprocess
import importlib.util
import time
import os

# ============================================================================
# FUNCIONES DE INSTALACIÓN DE DEPENDENCIAS                                  ||
# ============================================================================

def verificar_e_instalar_dependencias():
    """Verifica e instala dependencias necesarias."""
    
    dependencias = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'tensorflow': 'tensorflow'
    }
    
    faltantes = []
    
    print("="*60)
    print("VERIFICANDO DEPENDENCIAS")
    print("="*60)
    
    for nombre_modulo, nombre_paquete in dependencias.items():
        spec = importlib.util.find_spec(nombre_modulo)
        if spec is None:
            print(f"{nombre_modulo} no encontrado")
            faltantes.append(nombre_paquete)
        else:
            print(f"{nombre_modulo} encontrado")
    
    if faltantes:
        print(f"\nFaltan {len(faltantes)} dependencia(s): {', '.join(faltantes)}")
        respuesta = input("\n¿Desea instalarlas automáticamente? (s/n): ").lower()
        
        if respuesta == 's':
            print("\nInstalando dependencias... Esto puede tomar unos minutos.\n")
            try:
                # Instalar pip si no está disponible
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
                
                # Instalar todas las dependencias faltantes
                comando = [sys.executable, "-m", "pip", "install"] + faltantes
                subprocess.check_call(comando)
                print("\nDependencias instaladas exitosamente!")
                
                # Reiniciar el script para cargar las nuevas dependencias
                print("\nReiniciando la aplicación...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
            except subprocess.CalledProcessError as e:
                print(f"\nError durante la instalación: {e}")
                print("\nPor favor, instale manualmente con:")
                print(f"pip install {' '.join(faltantes)}")
                return False
        else:
            print("\nDependencias faltantes. Instale manualmente con:")
            print(f"pip install {' '.join(faltantes)}")
            return False
    
    print("\nTodas las dependencias están instaladas!")
    return True

# ============================================================================
# IMPORTACIONES CONDICIONALES                                               ||
# ============================================================================

def importar_modulos():
    """Importa todos los módulos necesarios de forma condicional."""
    
    modulos = {}
    
    # Intentar importar numpy
    try:
        import numpy as np
        modulos['np'] = np
        print("NumPy importado correctamente")
    except ImportError:
        print("Error: NumPy no está instalado")
        return None
    
    # Intentar importar matplotlib
    try:
        import matplotlib.pyplot as plt
        modulos['plt'] = plt
        print("Matplotlib importado correctamente")
    except ImportError:
        print("Error: Matplotlib no está instalado")
        return None
    
    # Intentar importar scipy
    try:
        from scipy.signal import correlate2d
        modulos['correlate2d'] = correlate2d
        print("SciPy importado correctamente")
    except ImportError:
        print("Error: SciPy no está instalado")
        return None
    
    # Intentar importar tensorflow (opcional para algunas funciones)
    try:
        import tensorflow as tf
        modulos['tf'] = tf
        print("TensorFlow importado correctamente")
    except ImportError:
        print("TensorFlow no encontrado (algunas funciones estarán limitadas)")
        modulos['tf'] = None
    
    return modulos

# ============================================================================
# VERIFICACIÓN INICIAL                                                      ||
# ============================================================================

# Verificar e instalar dependencias si es necesario
if not verificar_e_instalar_dependencias():
    sys.exit(1)

# Importar módulos
print("\n" + "="*60)
print("IMPORTANDO MÓDULOS")
print("="*60)
modulos = importar_modulos()

if modulos is None:
    print("\nNo se pudieron importar todos los módulos necesarios")
    sys.exit(1)

# Extraer módulos importados
np = modulos['np']
plt = modulos['plt']
correlate2d = modulos['correlate2d']
tf = modulos['tf']

# Configuración para reproducibilidad
if tf:
    tf.random.set_seed(42)
np.random.seed(42)

# ============================================================================
# 1. IMPLEMENTACIÓN MANUAL DE CONVOLUCIÓN                                   ||
# ============================================================================

def convolucion_manual(imagen, filtro, padding='same', stride=1):
    """
    Implementa manualmente la operación de convolución 2D.
    
    Args:
        imagen: Array 2D de entrada
        filtro: Kernel/filtro 2D
        padding: 'same' (mantiene dimensiones) o 'valid' (sin padding)
        stride: Paso de la convolución
        
    Returns:
        Resultado de la convolución
    """
    # Dimensiones
    img_h, img_w = imagen.shape
    filtro_h, filtro_w = filtro.shape
    
    # Calcular padding si es necesario
    if padding == 'same':
        pad_h = (filtro_h - 1) // 2
        pad_w = (filtro_w - 1) // 2
        imagen_padded = np.pad(imagen, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    else:  # 'valid'
        pad_h, pad_w = 0, 0
        imagen_padded = imagen
    
    # Calcular dimensiones de salida
    out_h = (img_h + 2*pad_h - filtro_h) // stride + 1
    out_w = (img_w + 2*pad_w - filtro_w) // stride + 1
    
    # Inicializar matriz de salida
    salida = np.zeros((out_h, out_w))
    
    # Realizar convolución
    for i in range(0, out_h):
        for j in range(0, out_w):
            # Extraer región de la imagen
            region = imagen_padded[i*stride:i*stride+filtro_h, 
                                  j*stride:j*stride+filtro_w]
            # Aplicar filtro (multiplicación elemento a elemento y suma)
            salida[i, j] = np.sum(region * filtro)
    
    return salida

# ============================================================================
# 2. IMPLEMENTACIÓN MANUAL DE MAX POOLING                                   ||
# ============================================================================

def max_pooling_manual(imagen, pool_size=2, stride=2):
    """
    Implementa manualmente la operación de max pooling 2D.
    
    Args:
        imagen: Array 2D de entrada
        pool_size: Tamaño de la ventana de pooling
        stride: Paso del pooling
        
    Returns:
        Resultado del max pooling
    """
    # Dimensiones
    img_h, img_w = imagen.shape
    
    # Calcular dimensiones de salida
    out_h = (img_h - pool_size) // stride + 1
    out_w = (img_w - pool_size) // stride + 1
    
    # Inicializar matriz de salida
    salida = np.zeros((out_h, out_w))
    
    # Aplicar max pooling
    for i in range(out_h):
        for j in range(out_w):
            # Extraer región
            region = imagen[i*stride:i*stride+pool_size, 
                           j*stride:j*stride+pool_size]
            # Tomar el valor máximo
            salida[i, j] = np.max(region)
    
    return salida

# ============================================================================
# 3. DEMOSTRACIÓN DE OPERACIONES MANUALES                                   ||
# ============================================================================

def demostrar_operaciones_manuales():
    """Demuestra las operaciones manuales de convolución y pooling."""
    print("="*60)
    print("DEMOSTRACIÓN DE OPERACIONES MANUALES")
    print("="*60)
    
    # Crear una imagen de ejemplo (5x5)
    imagen_ejemplo = np.array([
        [1, 2, 3, 0, 1],
        [4, 5, 6, 1, 2],
        [7, 8, 9, 2, 3],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5]
    ])
    
    # Crear un filtro de ejemplo (3x3) - detector de bordes
    filtro_ejemplo = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    
    print("\n1. Imagen de entrada (5x5):")
    print(imagen_ejemplo)
    
    print("\n2. Filtro (3x3) - detector de bordes:")
    print(filtro_ejemplo)
    
    # Aplicar convolución manual
    resultado_convolucion = convolucion_manual(imagen_ejemplo, filtro_ejemplo, padding='same')
    
    print("\n3. Resultado de convolución manual (5x5):")
    print(resultado_convolucion)
    
    # Aplicar max pooling manual
    resultado_pooling = max_pooling_manual(imagen_ejemplo, pool_size=2, stride=2)
    
    print("\n4. Resultado de max pooling manual (2x2):")
    print(resultado_pooling)
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].imshow(imagen_ejemplo, cmap='gray')
    axes[0, 0].set_title('Imagen original (5x5)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(filtro_ejemplo, cmap='gray')
    axes[0, 1].set_title('Filtro (3x3)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(resultado_convolucion, cmap='gray')
    axes[1, 0].set_title('Resultado convolución (5x5)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(resultado_pooling, cmap='gray')
    axes[1, 1].set_title('Resultado max pooling (2x2)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return imagen_ejemplo, filtro_ejemplo, resultado_convolucion, resultado_pooling

# ============================================================================
# 4. CNN CON TENSORFLOW (SOLO SI ESTÁ DISPONIBLE)                           ||
# ============================================================================

def crear_modelo_cnn():
    """
    Crea un modelo de CNN según la arquitectura especificada.
    Solo funciona si TensorFlow está instalado.
    """
    if tf is None:
        print(" TensorFlow no está instalado. No se puede crear el modelo CNN.")
        print("   Instale TensorFlow con: pip install tensorflow")
        return None
    
    try:
        from tensorflow.keras import layers, models
        
        modelo = models.Sequential([
            # Capa 1: Convolucional
               layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                 input_shape=(28, 28, 1), 
                    name='conv1'),
                        # Capa 2: Max Pooling
                            layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
            # Capa 3: Convolucional
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', 
            name='conv2'),
            # Capa 4: Max Pooling
            layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
            # Capa 5: Flatten
            layers.Flatten(name='flatten'),
            # Capa 6: Densa
            layers.Dense(128, activation='relu', name='dense1'),
            # Capa 7: Dropout
            layers.Dropout(0.5, name='dropout'),
            # Capa 8: Salida
            layers.Dense(10, activation='softmax', name='output')
        ])
        
        return modelo
    except Exception as e:
        print(f"Error al crear el modelo: {e}")
        return None

def entrenar_modelo_mnist(epochs=5):
    """
    Entrena el modelo CNN en el dataset MNIST.
    Solo funciona si TensorFlow está instalado.
    """
    if tf is None:
        print("TensorFlow no está instalado.")
        return None, None, None
    
    print("="*60)
    print("ENTRENAMIENTO DE CNN EN DATASET MNIST")
    print("="*60)
    
    try:
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        
        # Cargar dataset MNIST
        print("\n1. Cargando dataset MNIST...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Preprocesamiento
        print("2. Preprocesando datos...")
        
        # Normalizar imágenes a [0, 1]
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        # Añadir dimensión del canal
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        
        # Convertir etiquetas a one-hot encoding
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        print(f"   - Conjunto de entrenamiento: {x_train.shape[0]} muestras")
        print(f"   - Conjunto de prueba: {x_test.shape[0]} muestras")
        
        # Creacion del modelo
        print("3. Creando modelo CNN...")
        modelo = crear_modelo_cnn()
        
        if modelo is None:
            return None, None, None
        
        # Compilar el modelo
        modelo.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Resumen del modelo
        print("\n4. Resumen de la arquitectura del modelo:")
        modelo.summary()
        
        # comenzamos a entrenar el modelo
        print(f"\n5. Entrenando modelo ({epochs} épocas)...")
        inicio_entrenamiento = time.time()
        
        historia = modelo.fit(
            x_train, y_train,
            batch_size=128,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        
        tiempo_entrenamiento = time.time() - inicio_entrenamiento
        print(f"   Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")
        
        # Aqui evaluamos el modelo
        print("\n6. Evaluando modelo en conjunto de prueba...")
        test_loss, test_accuracy = modelo.evaluate(x_test, y_test, verbose=0)
        print(f"   Precisión en datos de prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return modelo, historia, (x_test, y_test)
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return None, None, None

def visualizar_resultados(modelo, historia, x_test, y_test):
    """
    Visualiza los resultados del entrenamiento y predicciones.
    """
    if modelo is None or historia is None:
        print("No hay modelo o historial para visualizar")
        return None, None
    
    # Gráficas de precisión y pérdida
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(historia.history.get('accuracy', []), label='Entrenamiento')
    plt.plot(historia.history.get('val_accuracy', []), label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(historia.history.get('loss', []), label='Entrenamiento')
    plt.plot(historia.history.get('val_loss', []), label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualizar algunas predicciones
    print("\n7. Visualizando algunas predicciones...")
    
    try:
        y_pred = modelo.predict(x_test[:12])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test[:12], axis=1)
        
        fig, axes = plt.subplots(3, 4, figsize=(10, 8))
        axes = axes.flatten()
        
        for i in range(12):
            ax = axes[i]
            ax.imshow(x_test[i].squeeze(), cmap='gray')
            color = 'green' if y_pred_classes[i] == y_true_classes[i] else 'red'
            ax.set_title(f'Real: {y_true_classes[i]}\nPred: {y_pred_classes[i]}', color=color)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return y_pred_classes, y_true_classes
        
    except Exception as e:
        print(f"Error al visualizar predicciones: {e}")
        return None, None

# ============================================================================
# 5. MENÚ PRINCIPAL                                                         ||
# ============================================================================

def mostrar_menu():
    """Muestra el menú principal de la aplicación."""
    print("\n" + "="*60)
    print("APLICACIÓN PRÁCTICA DE CNN PARA CLASIFICACIÓN DE DÍGITOS")
    print("="*60)
    print("\nOpciones disponibles:")
    print("1. Demostrar operaciones manuales (convolución y pooling)")
    
    if tf is not None:
        print("2. Entrenar y evaluar CNN en MNIST (2-3 épocas)")
        print("3. Entrenar y evaluar CNN en MNIST (5 épocas)")
        print("4. Visualizar operaciones matriciales en CNN")
        print("5. Salir")
        max_opcion = 5
    else:
        print("2. [DESHABILITADO] TensorFlow no está instalado")
        print("3. [DESHABILITADO] TensorFlow no está instalado")
        print("4. Visualizar operaciones matriciales en CNN")
        print("5. Salir")
        max_opcion = 5
    
    try:
        opcion = int(input(f"\nSeleccione una opción (1-{max_opcion}): "))
        return opcion
    except ValueError:
        print("Por favor, ingrese un número válido.")
        return 0

def explicar_operaciones_matriciales():
    """Explica las operaciones matriciales implementadas en la CNN."""
    print("\n" + "="*60)
    print("OPERACIONES MATRICIALES EN CNN")
    print("="*60)
    
    print("\n1. CONVOLUCIÓN 2D:")
    print("   - Operación: Suma de productos entre filtro y región de imagen")
    print("   - Implementación manual: Bucles anidados con np.sum(region * filtro)")
    print("   - En Keras: layers.Conv2D con multiplicación matricial optimizada")
    
    print("\n2. MAX POOLING 2D:")
    print("   - Operación: Extracción del valor máximo de submatrices")
    print("   - Implementación manual: Bucles con np.max(region)")
    print("   - En Keras: layers.MaxPooling2D con operaciones vectorizadas")
    
    print("\n3. MULTIPLICACIÓN MATRICIAL (CAPAS DENSAS):")
    print("   - Fórmula: Y = X × W + b")
    print("   - X: Vector de entrada")
    print("   - W: Matriz de pesos")
    print("   - b: Vector de sesgo")
    print("   - Y: Salida de la capa")
    
    print("\n4. TRANSFORMACIONES DE FORMA:")
    print("   - Flatten: Convierte matriz 3D en vector 1D")
    print("   - Ejemplo: (14, 14, 64) → (12544,)")
    
    print("\n5. OPERACIONES DE ACTIVACIÓN:")
    print("   - ReLU: max(0, x) - Introduce no linealidad")
    print("   - Softmax: exp(x_i)/∑exp(x_j) - Probabilidades normalizadas")
    
    print("\n6. OPERACIÓN DE DROPOUT:")
    print("   - Regularización: Apaga el 50% de neuronas aleatoriamente")
    print("   - Previene overfitting durante el entrenamiento")
    
    input("\nPresione Enter para continuar...")

# ============================================================================
# 6. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación."""
    print("\n" + "="*60)
    print("BIENVENIDO A LA APLICACIÓN PRÁCTICA DE CNN")
    print("="*60)
    print("\nEsta aplicación implementa:")
    print("   Convolución manual")
    print("   Max Pooling manual")
    
    if tf is not None:
        print("   CNN completa para MNIST (~95-98% precisión)")
    else:
        print("    CNN para MNIST (requiere TensorFlow)")
    
    print("\n" + "="*60)
    
    modelo_entrenado = None
    historia = None
    datos_prueba = None
    
    while True:
        opcion = mostrar_menu()
        
        if opcion == 1:
            # Demostrar operaciones manuales
            demostrar_operaciones_manuales()
            
        elif opcion == 2 and tf is not None:
            # Entrenar modelo con 2-3 épocas
            modelo_entrenado, historia, datos_prueba = entrenar_modelo_mnist(epochs=2)
            if modelo_entrenado is not None:
                visualizar_resultados(modelo_entrenado, historia, *datos_prueba)
            
        elif opcion == 3 and tf is not None:
            # Entrenar modelo con 5 épocas
            modelo_entrenado, historia, datos_prueba = entrenar_modelo_mnist(epochs=5)
            if modelo_entrenado is not None:
                visualizar_resultados(modelo_entrenado, historia, *datos_prueba)
            
        elif opcion == 4:
            # Explicar operaciones matriciales
            explicar_operaciones_matriciales()
            
        elif opcion == 5:
            # Salir
            print("\n¡Gracias por usar la aplicación!")
            print("="*60)
            break
            
        elif tf is None and opcion in [2, 3]:
            print("\n Esta opción requiere TensorFlow.")
            print("   Instálelo con: pip install tensorflow")
            print("   O ejecute el script nuevamente para instalación automática.")
            
        else:
            print("\n Opción no válida. Por favor, seleccione una opción del menú.")

# ============================================================================
# 7. EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Aplicación interrumpida por el usuario.")
    except Exception as e:
        print(f"\n Error inesperado: {e}")
        print("\nPosibles soluciones:")
        print("1. Asegúrese de tener todas las dependencias instaladas")
        print("2. Ejecute el script nuevamente para instalación automática")
        print("3. Si el problema persiste, instale manualmente:")
        print("   pip install numpy matplotlib scipy tensorflow")
    
    input("\nPresione Enter para salir...")