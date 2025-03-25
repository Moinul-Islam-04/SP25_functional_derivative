import sympy as sp

def compute_functional_derivative(integrand, *args):
    """
    Compute functional derivative for both single and double integral expressions.
    
    For single integrals:
    Args:
        integrand: The integrand containing one variable
        var: Integration variable (e.g., r)
        func: Function of the variable (e.g., rho(r))
        delta: Perturbation function
    
    For double integrals:
    Args:
        integrand: The integrand containing two variables
        var1, var2: Integration variables (e.g., r1, r2)
        func1, func2: Functions of the respective variables
        delta1, delta2: Perturbation functions
    
    Returns:
        sympy.Expr: The functional derivative after integration
    """
    epsilon = sp.Symbol('epsilon')
    
    # Determine if it's a single or double integral based on number of arguments
    is_single_integral = len(args) == 3
    
    if is_single_integral:
        var, func, delta = args
        # Handle single integral case
        perturbed = func + epsilon*delta
        integrand_with_delta = integrand.subs(func, perturbed)
        derivative = sp.diff(integrand_with_delta, epsilon).subs(epsilon, 0)
        result = derivative
        
    else:
        # Double integral case
        var1, var2, func1, func2, delta1, delta2 = args
        
        # Check if this is the general case (with powers a and b)
        a, b = sp.symbols('a b')
        is_general_case = a in integrand.free_symbols or b in integrand.free_symbols
        
        if is_general_case:
            # First variation with respect to func1
            perturbed1 = func1 + epsilon*delta1
            perturbed_integrand1 = integrand.subs(func1, perturbed1)
            derivative1 = sp.diff(perturbed_integrand1, epsilon).subs(epsilon, 0)
            
            # Second variation with respect to func2
            perturbed2 = func2 + epsilon*delta2
            perturbed_integrand2 = integrand.subs(func2, perturbed2)
            derivative2 = sp.diff(perturbed_integrand2, epsilon).subs(epsilon, 0)
            
            result = derivative1 + derivative2
            
        else:
            # Handle specific cases (e.g., Hartree potential)
            perturbed = func1 + epsilon*delta1
            integrand_with_delta = integrand.subs(func1, perturbed)
            derivative = sp.diff(integrand_with_delta, epsilon).subs(epsilon, 0)
            result = sp.Integral(derivative, (var2, -sp.oo, sp.oo))
    
    return result.simplify()

# Test functions
def test_hartree():
    r1, r2 = sp.symbols('r1 r2')
    rho1 = sp.Function('rho')(r1)
    rho2 = sp.Function('rho')(r2)
    delta_rho1 = sp.Function('delta_rho')(r1)
    delta_rho2 = sp.Function('delta_rho')(r2)
    
    hartree_integrand = rho1 * rho2 / sp.Abs(r1 - r2)
    
    result = compute_functional_derivative(
        hartree_integrand,
        r1, r2,
        rho1, rho2,
        delta_rho1, delta_rho2
    )
    return result

def test_general():
    r1, r2 = sp.symbols('r1 r2')
    a, b = sp.symbols('a b')
    rho1 = sp.Function('rho')(r1)
    rho2 = sp.Function('rho')(r2)
    delta_rho1 = sp.Function('delta_rho')(r1)
    delta_rho2 = sp.Function('delta_rho')(r2)
    W = sp.Function('W')(sp.Abs(r1 - r2))
    
    test_integrand = (rho1**a) * W * (rho2**b)
    
    result = compute_functional_derivative(
        test_integrand,
        r1, r2,
        rho1, rho2,
        delta_rho1, delta_rho2
    )
    return result

def test_XC():
    r = sp.symbols('r')
    rho = sp.Function('rho')(r)
    delta_rho = sp.Function('delta_rho')(r)
    C_x = sp.symbols('C_x')
    
    integrand_xc = C_x * rho**(4/3)
    result = compute_functional_derivative(
        integrand_xc,
        r, rho, delta_rho
    )
    print("Functional Derivative (XC Energy):")
    print(result)

def test_TF():
    r = sp.symbols('r')
    rho = sp.Function('rho')(r)
    delta_rho = sp.Function('delta_rho')(r)
    C_TF = sp.symbols('C_TF')
    
    integrand_tf = C_TF * rho**(5/3)
    result = compute_functional_derivative(
        integrand_tf,
        r, rho, delta_rho
    )
    print("\nFunctional Derivative (Thomas–Fermi Kinetic Energy):")
    print(result)



def WinKx():
    x, y = sp.symbols('x y')
    f = sp.Function('f')
    g = sp.Function('g')
    expr = f(g(x))
    dydx = sp.diff(expr, x)
    print(dydx)


# Run all tests
#print("Hartree potential result:")
#print(test_hartree())
#print("\nGeneral case result:")
#print(test_general())
#test_XC()
#test_TF()

WinKx()