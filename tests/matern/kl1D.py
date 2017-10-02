from dolfin import * #@UnusedWildImport
import logging
import numpy as np
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_level(WARNING)

def kl1D(y, N, exponent,order=1):
    deg = order
    mesh = UnitIntervalMesh(int(N))
    V = FunctionSpace(mesh, 'CG', deg)
    bc = DirichletBC(V,Constant(0.0), lambda x,on_boundary: on_boundary)
    output = np.zeros((y.shape[0], 1))
    ut = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    L = f * v * dx
    d = y.shape[1]
    expression_string=diffusion_coefficient(d,exponent)
    for j in range(y.shape[0]):
        # Compute solution
        arg_dict = { 'y%d' % (i+1): v for i, v in enumerate(y[j,:])}
        arg_dict['degree']=2*deg
        D = Expression(expression_string, **arg_dict)
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
        integrand = u * dx
        a = assemble(integrand)
        # print(a)
        output[j] = a
    return output

def diffusion_coefficient(d,exponent):
    expression='exp( 0 '
    for dd in range(d):
        dd=dd+1
        expression+='+'
        expression+='pow({},-{})*y{}*'.format(dd,exponent,dd)
        if dd%2==0:
            expression +='sin({}*pi*x[0])'.format(dd)
        else:
            expression +='cos({}*pi*x[0])'.format(dd)
    expression+=')'
    return expression

if __name__=='__main__':
    from sat.pde.kl1D import kl1D
    kl1D(np.array([[1,1],[1,1]]),32,order=1)
