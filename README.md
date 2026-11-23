# mamba-collapse

Proyecto de Métodos Diferenciales de la IA: colapso en Mamba y modelos de espacio de estados selectivos.

Referencia principal: Vo, Pham, Tong y Nguyen (2025), Demystifying the token dynamics of deep selective state space models, ICLR (arXiv 2410.03292).

Objetivo: formular la dinámica de tokens de un SSM selectivo como un sistema continuo, comprobar numéricamente cuándo los tokens convergen a cero o divergen, y estudiar cómo afectan la dimensión de estado, la parametrización y la longitud de la secuencia. Después comparar con el colapso en self-attention.

## Estructura

- src/: código compartido (simulador base del SSM selectivo)
- experiments/: un script por experimento, cada uno ejecutable por separado
- figures/: figuras generadas, que se usan en el documento y en la charla
- paper/: notas de notación y enlace al Overleaf

## Experimentos

| Script | Contenido | Responsable |
|---|---|---|
| sim_dimension_estado.py | efecto de la dimensión de estado | Isabel |
| sim_parametrizacion.py | efecto de la parametrización (signo de mu, Delta, a) | Raquel |
| sim_longitud_secuencia.py | longitud de secuencia y memoria de largo contexto | Raquel |

## Cómo trabajar

- La estructura y el README se tocan directamente en main.
- Todo lo demás va en una rama propia (por ejemplo feat/sim-parametrizacion) y se mezcla con main mediante pull request.
- src/ssm.py es compartido: avisar antes de cambiar su interfaz.
- Cada experimento guarda sus figuras en figures/.

## Instalación

    pip install -r requirements.txt
