import textwrap

import numpy as np

def write_visit_input(
        stem,
        grid,
        size,
        invert,
        energy_scale,
        save_dir="."):

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


def write_griday_input(
        stem,
        grid_tuple,
        size,
        invert,
        energy_scale,
        cell_length_scale,
        save_dir="."):
        #cell_angle_scale=[0.0, 180.0],

    cell = grid_tuple[0]

    cmin = cell_length_scale[0]
    cmax = cell_length_scale[1]

    cell[:3] = (cmax - cmin)*cell[:3] + cmin

    #amin = cell_angle_scale[0]
    #amax = cell_angle_scale[1]
    #cell[3:] = (amax - amin)*cell[3:] + amin

    cell = list(cell)

    lower = energy_scale[0]
    upper = energy_scale[1]

    data = np.array(grid_tuple[1].reshape([-1]))

    if invert:
        data = 1.0 - data

    data = (upper-lower)*data + lower

    # Make file name
    grid = save_dir + "/" + stem + ".grid"
    griddata = grid + "data"

    # Write header file
    with open(grid, "w") as gridfile:
        gridfile.write(textwrap.dedent("""\
            CELL_PARAMETERS {} {} {}
            CELL_ANGLES {} {} {}
            GRID_NUMBERS {size} {size} {size}""".format(
                cell[0], cell[1], cell[2],
                90, 90, 90,
                #cell[3], cell[4], cell[5],
                size=size))
            )

    data.tofile(griddata)
