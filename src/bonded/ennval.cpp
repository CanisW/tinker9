#include "ff/energy.h"
#include "ff/evalence.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "iostream"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <string>
#include <cuda_runtime.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>


namespace tinker {

PyObject *py_amoeba_nn;
PyObject *py_nn_analyze;

void ennvalenceData(RcOp op)
{
   // if (not use(Potent::NNVAL))
   //    return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {

      // if (rc_a)
         bufferDeallocate(rc_flag, ennval, vir_ennval, dennval_x, dennval_y, dennval_z);
      // ennval = nullptr;
      // vir_ennval = nullptr;
      // dennval_x = nullptr;
      // dennval_y = nullptr;
      // dennval_z = nullptr;
      Py_XDECREF(py_nn_analyze);
      Py_DECREF(py_amoeba_nn);
   }

   if (op & RcOp::ALLOC) {

      // ennval = eng_buf;
      // vir_ennval = vir_buf;
      // dennval_x = gx;
      // dennval_y = gy;
      // dennval_z = gz;
      // if (rc_a)
         bufferAllocate(rc_flag, &ennval, &vir_ennval, &dennval_x, &dennval_y, &dennval_z);
      // wchar_t *path, *newpath;
      // path=Py_GetPath();
      // newpath=new wchar_t[wcslen(path)+50];
      // wcscpy(newpath, path);
      // std::cout << "path: " << path << ", " << wcslen(path) << std::endl;
      // std::cout << "newpath: " << newpath << ", " << wcslen(newpath) << std::endl;
      // wcscat(newpath, L":/home/yw24267/work/DLFF/tinker9/ext/");
      // std::cout << "newpath2: " << newpath << ", " << wcslen(newpath) << std::endl;
      // PySys_SetPath(newpath);
      // free(newpath);

      Py_Initialize();

      py_amoeba_nn = PyImport_ImportModule("amoeba_nn.utils.tinker9_interface");
      // std::cout << "py_amoeba_nn imported "<< std::endl;
      if (py_amoeba_nn != NULL) {
         // std::string py_func_name = "utils.tinker9_interface.nn_analyze";
         // py_nn_analyze = PyObject_GetAttrString(py_amoeba_nn, py_func_name.c_str());
         py_nn_analyze = PyObject_GetAttrString(py_amoeba_nn, "nn_analyze");
         if (py_nn_analyze && PyCallable_Check(py_nn_analyze)) {
            // std::cout << "py_nn_analyze callable "<< std::endl;
         } else {
            if (PyErr_Occurred())
                  PyErr_Print();
            fprintf(stderr, "Cannot find energy function for AMOEBA+NN.\n");
         }

      } else {
         PyErr_Print();
         fprintf(stderr, "Failed to load AMOEBA+NN\n");
      }
   }

   if (op & RcOp::INIT) {}
}


void ennvalence(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   // if (rc_a) {
      // why only do this when rc_a?
      zeroOnHost(energy_ennval, vir_ennval);
   // }

   // std::cout << "num of atoms: " << n << std::endl;
   // std::cout << "atomic: " << sizeof(tinker::atomid::atomic) << ", " << tinker::atomid::atomic[0] << ", " << tinker::atomid::atomic[100] << std::endl;
   // std::cout << "tinker::atoms::x: " << sizeof(tinker::atoms::x) << ", " << tinker::atoms::x[0] << ", " << tinker::atoms::x[100] << std::endl;

   real *x_cpu, *y_cpu, *z_cpu;
   x_cpu = new real[n];
   y_cpu = new real[n];
   z_cpu = new real[n];
   cudaDeviceSynchronize();
   cudaMemcpy(x_cpu, x, n*sizeof(real), cudaMemcpyDeviceToHost);
   cudaMemcpy(y_cpu, y, n*sizeof(real), cudaMemcpyDeviceToHost);
   cudaMemcpy(z_cpu, z, n*sizeof(real), cudaMemcpyDeviceToHost);

   PyObject *py_args, *py_values, *py_vi;

   if (py_amoeba_nn != NULL) {
      if (py_nn_analyze && PyCallable_Check(py_nn_analyze)) {
         // std::cout << "py_nn_analyze callable "<< std::endl;
         py_args = PyTuple_New(2);

         py_values = PyList_New(n);
         for (int i = 0; i < n; ++i) {
            PyList_SetItem(py_values, i, PyLong_FromLong(atomid::atomic[i]));
         }
         PyTuple_SetItem(py_args, 0, py_values);
         // std::cout << "py_args[0] set "<< std::endl;

         py_values = PyList_New(n);
         for (int i = 0; i < n; ++i) {
            py_vi = PyList_New(3);
            PyList_SetItem(py_vi, 0, PyFloat_FromDouble(x_cpu[i]));
            PyList_SetItem(py_vi, 1, PyFloat_FromDouble(y_cpu[i]));
            PyList_SetItem(py_vi, 2, PyFloat_FromDouble(z_cpu[i]));
            PyList_SetItem(py_values, i, py_vi);
         }
         PyTuple_SetItem(py_args, 1, py_values);
         // std::cout << "py_args[1] set "<< std::endl;
         // std::cout << "Ref count4: " << Py_REFCNT(py_values) << std::endl;         

         py_values = PyObject_CallObject(py_nn_analyze, py_args);
         // std::cout << "py_nn_analyze called "<< std::endl;
         Py_DECREF(py_args);
         if (py_values != NULL) {
            // std::cout << "py_values not NULL "<< std::endl;
            // std::cout << "py_values size: " << PyTuple_Size(py_values) << std::endl;
            energy_ennval = PyFloat_AsDouble(PyTuple_GetItem(py_values, 0));
            // std::cout << "Result of call: Energy: " << energy_ennval << std::endl;
            // std::cout << "dennval_x: " << typeid(dennval_x).name() << std::endl;
            grad_prec *dx_cpu, *dy_cpu, *dz_cpu;
            dx_cpu = new grad_prec[n];
            dy_cpu = new grad_prec[n];
            dz_cpu = new grad_prec[n];
            for (int i = 0; i < n; ++i) {
               dx_cpu[i] = PyFloat_AsDouble(PyTuple_GetItem(PyTuple_GetItem(py_values, 1), i));
               dy_cpu[i] = PyFloat_AsDouble(PyTuple_GetItem(PyTuple_GetItem(py_values, 2), i));
               dz_cpu[i] = PyFloat_AsDouble(PyTuple_GetItem(PyTuple_GetItem(py_values, 3), i));
            }
            cudaMemcpy(dennval_x, dx_cpu, n*sizeof(grad_prec), cudaMemcpyHostToDevice);
            cudaMemcpy(dennval_y, dy_cpu, n*sizeof(grad_prec), cudaMemcpyHostToDevice);
            cudaMemcpy(dennval_z, dz_cpu, n*sizeof(grad_prec), cudaMemcpyHostToDevice);
            // std::cout << "Result of call: Gradient: " << dx_cpu[0] << ", " << dy_cpu[0] << ", " << dz_cpu[0] << std::endl;
            // std::cout << "Result of call: Gradient: " << dx_cpu[10] << ", " << dy_cpu[10] << ", " << dz_cpu[10] << std::endl;
            delete[] dx_cpu;
            delete[] dy_cpu;
            delete[] dz_cpu;
            Py_DECREF(py_values);
         }
         else {
            PyErr_Print();
            // PyObject *ptype, *pvalue, *ptraceback;
            // PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            //pvalue contains error message
            //ptraceback contains stack snapshot and many other information
            //(see python traceback structure)

            // //Get error message
            // const char *pStrErrorMessage = PyUnicode_AsUTF8(pvalue);
            // std::cout << "value: " << pStrErrorMessage << std::endl;
            // const char *pStrErrorMessage2 = PyUnicode_AsUTF8(ptype);
            // std::cout << "type: " << pStrErrorMessage2 << std::endl;
            // const char *pStrErrorMessage3 = PyUnicode_AsUTF8(ptraceback);
            // std::cout << "traceback: " << pStrErrorMessage3 << std::endl;
            fprintf(stderr, "AMOEBA+NN energy function call failed\n");
            exit (1);
         }
      }
   }
   delete[] x_cpu;
   delete[] y_cpu;
   delete[] z_cpu;

   // if (rc_a) {
      if (do_e) {
         // energy_ennval = energyReduce(ennval);
         energy_valence += energy_ennval;
      }
      if (do_v) {
         // virialReduce(virial_ennval, vir_ennval);
         // for (int iv = 0; iv < 9; ++iv)
         //    virial_valence[iv] += virial_eb[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, dennval_x, dennval_y, dennval_z);
   // }
}
}
