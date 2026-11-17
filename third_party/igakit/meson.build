project(
  'igakit',
  'c', 'fortran',
  version: '0.1.0',
  license: 'BSD-2-Clause',
  license_files: ['LICENSE.rst'],
  meson_version: '>=1.1.0',
)

fs = import('fs')
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true,
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true,
).stdout().strip()

np_dep = declare_dependency(
  # include_directories: [incdir_numpy, incdir_f2py],
  compile_args: [f'-I@incdir_numpy@', f'-I@incdir_f2py@'],
)

f2py = [
  py, '-m', 'numpy.f2py',
  '--build-dir', '@OUTDIR@',
  '@INPUT@',
]

fc = meson.get_compiler('fortran')
cc = meson.get_compiler('c')
add_project_arguments(
  fc.get_supported_arguments(
    #'-Wno-unused-dummy-argument',
  ),
  language : 'fortran',
)
add_project_arguments(
  cc.get_supported_arguments(
    '-Wno-misleading-indentation',
  ),
  language : 'c',
)

ext_name = 'igalib'
ext_fort_srcs = ['igakit' / ext_name + '.f90']
ext_f2py_srcs = custom_target(
  command: [f2py],
  input: ['igakit' / ext_name + '.pyf'],
  output: [
    ext_name + 'module.c',
    ext_name + '-f2pywrappers2.f90',
  ],
)

py.extension_module(
  ext_name, [ext_fort_srcs, ext_f2py_srcs],
  incdir_f2py / 'fortranobject.c',
  dependencies : [py_dep, np_dep],
  subdir: 'igakit',
  install : true
)

package = {
  'igakit': [
    '__init__.py',
    'cad.py',
    'io.py',
    'nurbs.py',
    'plot_mpl.py',
    'plot_myv.py',
    'plot_nul.py',
    'plot.py',
    'transform.py',
  ]
}
foreach pkg, src : package
  srcdir = ''
  subdir = join_paths(pkg.split('.'))
  sources = []
  foreach fn : src
    sources += srcdir / subdir / fn
  endforeach
  py.install_sources(
    sources,
    pure: false,
    subdir: subdir,
  )
endforeach
