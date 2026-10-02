class colors(object):
    black   = (0.00, 0.00, 0.00)
    grey    = (0.75, 0.75, 0.75)
    white   = (1.00, 1.00, 1.00)
    #
    red     = (1.00, 0.00, 0.00)
    green   = (0.00, 1.00, 0.00)
    blue    = (0.00, 0.00, 1.00)
    #
    yellow  = (1.00, 1.00, 0.00)
    cyan    = (0.00, 1.00, 1.00)
    magenta = (1.00, 0.00, 1.00)
    #
    orange  = (1.00, 0.50, 0.00)

class _namespace(object):
    def __getattr__(self, name):
        def func(*args, **kwargs): return None
        func.__name__ = name
        func.__doc__  = ''
        return func

nul = _namespace()

figure = nul.figure
gcf = nul.gcf
clf = nul.clf
close = nul.close
save = nul.savefig
show = nul.show

title = nul.title
xlabel = nul.xlabel
ylabel = nul.ylabel
zlabel = nul.zlabel
colorbar = nul.colorbar

points3d = nul.points3d
quiver3d = nul.quiver3d
line3d = nul.quiver3d
surf3d = nul.quiver3d

_resolution = { 1:64, 2:32 }
