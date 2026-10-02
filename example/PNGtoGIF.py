import imageio.v2, os
GIF = []
filepath = "examples/mpm/column_collapse/Newtonian/animation"
filenames = sorted((fn for fn in os.listdir(filepath) if fn.endswith('.png')))
for filename in filenames:
    GIF.append(imageio.v2.imread(filepath + "/" + filename))
imageio.mimsave(filepath + "/" + 'mpdem.gif', GIF, fps=10)
