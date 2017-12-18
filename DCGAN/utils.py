import textwrap

import numpy as np

def write_visit_input(
        stem,
        grid,
        size,
        save_dir=".",
        invert=True,
        energy_scale=[-6000, 6000]):

    lower = energy_scale[0]
    upper = energy_scale[1]

    data = np.array(grid.reshape([-1]))

    if invert:
        data = 1.0 - data

    data = (upper-lower)*data + lower

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
