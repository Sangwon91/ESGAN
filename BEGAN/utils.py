import textwrap

import numpy as np

def write_visit_input(stem, grid, size, energy_range, save_dir=".", invert=True):
    min_energy = energy_range[0]
    max_energy = energy_range[1]

    de = max_energy - min_energy

    # Transform to original structure.
    #data = (max_energy - de * 0.5*(1.0+grid)).reshape((-1))
    data = grid.reshape((-1))
    if invert:
        data = 1.0 - data
    data = de*data + min_energy
    #data = grid.reshape((-1))

    # Make file name
    bov = save_dir + "/" + stem + ".bov"
    times = stem + ".times"

    # Write header file
    with open(bov, "w") as bovfile:
        bovfile.write(textwrap.dedent("""\
            TIME: 1.000000
            DATA_FILE: {}
            DATA_SIZE:     {size} {size} {size}
            DATA_FORMAT: FLOAT
            VARIABLE: data
            DATA_ENDIAN: LITTLE
            CENTERING: nodal
            BRICK_ORIGIN:        0  0  0
            BRICK_SIZE:       {box} {box} {box}""".format(
            times, size=size, box=int(0.5*size + 1e-6))
        ))

    data.tofile(save_dir + "/" + times)
