from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dolfin import * #@UnusedWildImport
import logging
import numpy as np
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_level(WARNING)

def matern(y, N, nu, order=1,
           df_sig=0.5, df_L=1, qoi_x0=0.3, qoi_sig=1.):
    deg = order
    mesh = UnitIntervalMesh(int(N))
    V = FunctionSpace(mesh, 'CG', deg)
    bc = DirichletBC(V,Constant(0.0), lambda x,on_boundary: on_boundary)
    output = np.zeros((y.shape[0], 1))
    ut = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    L = f * v * dx
    d = y.shape[1]
    expression_string=diffusion_coefficient(d,nu)
    for j in range(y.shape[0]):
        # Compute solution
        arg_dict = { 'y%d' % (i+1): v for i, v in enumerate(y[j,:])}
        arg_dict['degree']=2*deg
        D = Expression(expression_string, sig=df_sig,
                       L=df_L, **arg_dict)
        a = dot(D * grad(ut), grad(v)) * dx
        u = Function(V)
        problem = LinearVariationalProblem(a, L, u, bc)
        solver  = LinearVariationalSolver(problem)
        solver.parameters['linear_solver'] = 'cg'
        solver.parameters['preconditioner'] = 'amg'
        cg_prm = solver.parameters['krylov_solver']
        #cg_prm['absolute_tolerance'] = 1E-25
        cg_prm['relative_tolerance'] = 1E-25
        #cg_prm['maximum_iterations'] = 10000
        solver.solve()
        g = Expression('exp(-0.5*pow(x[0]-pt, 2)/sig2)',
                       degree=2*deg, sig2=qoi_sig**2, pt=qoi_x0)
        integrand = np.power(2*(qoi_sig**2)*np.pi, -0.5) * g * u * dx
        a = assemble(integrand)
        # print(a)
        output[j] = a
    return output

def diffusion_coefficient(d,nu):
    expression='exp(0'
    for dd in range(1, d+1):
        k = dd // 2
        expression += '+sig*pow(2, {k0})*pow(pow({k}, 2)+1,-{exp})*y{dd}*'\
                      .format(k0=0.5 if k>0 else 0, k=k,
                              dd=dd, exp=0.5*(nu+0.5))
        if dd%2==0:
            expression +='sin({k}*pi*x[0]/L)'.format(k=k)
        else:
            expression +='cos({k}*pi*x[0]/L)'.format(k=k)
    expression+=')'
    return expression

if __name__=='__main__':
    # matern(np.array([[1,1],[1,1]]), 32, 3.5, order=1)
    output = matern(np.array([[1,1,1,1],[1,1,1,1]]), 32, 3.5, order=1)
    print(output[:, 0])
