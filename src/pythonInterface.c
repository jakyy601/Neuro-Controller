#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *hello(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyUnicode_FromString("Hello from C");
}

static PyMethodDef MyMethods[] = {
    {"hello", hello, METH_NOARGS, "Return a greeting string."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mymoduledef = {
    PyModuleDef_HEAD_INIT,
    "mymodule",              /* m_name */
    "Minimal C extension.",  /* m_doc */
    -1,                      /* m_size */
    MyMethods                /* m_methods */
};

PyMODINIT_FUNC PyInit_mymodule(void) {
    return PyModule_Create(&mymoduledef);
}
