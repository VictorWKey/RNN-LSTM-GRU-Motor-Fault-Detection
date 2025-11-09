"""
Script principal para ejecutar ambas tareas: clasificación y regresión.

Este script ejecuta secuencialmente:
1. Entrenamiento de modelos de clasificación
2. Entrenamiento de modelos de regresión

Uso:
    python run_all.py
    
    # Solo clasificación
    python run_all.py --classification-only
    
    # Solo regresión
    python run_all.py --regression-only
"""
import sys
import argparse
import subprocess
from datetime import datetime


def run_script(script_name, description):
    """
    Ejecuta un script de Python y maneja errores.
    
    Args:
        script_name: nombre del script a ejecutar
        description: descripción del script
    """
    print("\n" + "="*80)
    print(f"EJECUTANDO: {description}")
    print("="*80)
    print(f"Script: {script_name}")
    print(f"Hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    try:
        # Ejecutar script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        print("\n" + "="*80)
        print(f"✅ {description} COMPLETADO EXITOSAMENTE")
        print("="*80 + "\n")
        
        return True
    
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print(f"❌ ERROR EN: {description}")
        print("="*80)
        print(f"Código de salida: {e.returncode}")
        print("="*80 + "\n")
        
        return False
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Ejecución interrumpida por el usuario")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Ejecuta scripts de entrenamiento de la Práctica 02'
    )
    parser.add_argument(
        '--classification-only',
        action='store_true',
        help='Ejecutar solo entrenamiento de clasificación'
    )
    parser.add_argument(
        '--regression-only',
        action='store_true',
        help='Ejecutar solo entrenamiento de regresión'
    )
    
    args = parser.parse_args()
    
    # Banner inicial
    print("\n" + "="*80)
    print("PRÁCTICA 02 - EJECUCIÓN COMPLETA")
    print("Modelos Recurrentes para Señales de Motor")
    print("="*80)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    results = {}
    
    # Determinar qué ejecutar
    run_classification = not args.regression_only
    run_regression = not args.classification_only
    
    # Ejecutar clasificación
    if run_classification:
        print("\n📊 TAREA 1: CLASIFICACIÓN DE SEÑALES")
        results['classification'] = run_script(
            'train_classification.py',
            'Entrenamiento de Clasificación'
        )
        
        if not results['classification']:
            print("\n⚠️  Clasificación falló. ¿Deseas continuar con regresión? (s/n): ", end='')
            response = input().strip().lower()
            if response != 's':
                print("\n❌ Ejecución abortada")
                sys.exit(1)
    
    # Ejecutar regresión
    if run_regression:
        print("\n📈 TAREA 2: REGRESIÓN DE SERIES TEMPORALES")
        results['regression'] = run_script(
            'train_regression.py',
            'Entrenamiento de Regresión'
        )
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE EJECUCIÓN")
    print("="*80)
    
    if run_classification:
        status = "✅ EXITOSO" if results.get('classification', False) else "❌ FALLIDO"
        print(f"Clasificación: {status}")
    
    if run_regression:
        status = "✅ EXITOSO" if results.get('regression', False) else "❌ FALLIDO"
        print(f"Regresión: {status}")
    
    print("="*80)
    print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Verificar si todo fue exitoso
    all_successful = all(results.values())
    
    if all_successful:
        print("🎉 ¡TODOS LOS ENTRENAMIENTOS COMPLETADOS EXITOSAMENTE!")
        print("\n📁 Revisa los resultados en:")
        print("   - checkpoints/  (modelos guardados)")
        print("   - figures/      (gráficas)")
        print("   - results/      (tablas CSV y LaTeX)")
        sys.exit(0)
    else:
        print("⚠️  Algunos entrenamientos fallaron. Revisa los mensajes de error arriba.")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ejecución interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
