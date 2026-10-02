import imageio.v2, os
GIF = []
filepath = "."
filenames = sorted((fn for fn in os.listdir(filepath) if fn.endswith('.png')))
for filename in filenames:
    GIF.append(imageio.v2.imread(filepath + "/" + filename))
imageio.mimsave(filepath + "/" + 'multibody.gif', GIF, fps=10, loop=0)
