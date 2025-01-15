import sympy as sp

def compute_functional_derivative_taylor(functional, integrand, perturbation_variable, perturbation):
    """
    Compute the functional derivative using a Taylor expansion.
    
    Args:
        functional (sympy.Expr): The functional E[f], typically an integral.
        integrand (sympy.Expr): The integrand of the functional.
        perturbation_variable (sympy.Function): The function being varied (e.g., rho(r)).
        perturbation (sympy.Function): The perturbation variable (e.g., delta_rho(r)).
    
    Returns:
        sympy.Expr: The functional derivative.
    """
    perturbed_variable = perturbation_variable + perturbation
    perturbed_integrand = integrand.subs(perturbation_variable, perturbed_variable)
    functional_derivative = sp.diff(perturbed_integrand, perturbation).subs(perturbation, 0)
    return functional_derivative

def compute_double_integral_derivative(integrand, var1, var2, func1, func2, delta1, delta2):
    """
    Compute functional derivative for double integral expressions.
    
    Args:
        integrand (sympy.Expr): The integrand containing two variables
        var1, var2 (sympy.Symbol): Integration variables (e.g., r1, r2)
        func1, func2 (sympy.Function): Functions of the respective variables
        delta1, delta2 (sympy.Function): Perturbation functions
    
    Returns:
        sympy.Expr: The functional derivative after integration
    """
    # Perturb both functions
    perturbed_integrand = integrand.subs([
        (func1, func1 + delta1),
        (func2, func2 + delta2)
    ])
    
    # Expand and get first-order terms
    expanded = sp.expand(perturbed_integrand)
    
    # Take derivative with respect to first perturbation and evaluate at zero
    derivative = sp.diff(expanded, delta1).subs([
        (delta1, 0),
        (delta2, 0)
    ])
    
    # Integrate over second variable
    full_derivative = sp.Integral(derivative, (var2, -sp.oo, sp.oo))
    
    return full_derivative

# Symbols
r = sp.symbols('r')
r_prime = sp.symbols('r_prime')
rho = sp.Function('rho')(r)
rho_prime = sp.Function('rho')(r_prime)
delta_rho = sp.Function('delta_rho')(r)
delta_rho1 = sp.Function('delta_rho')(r_prime)
C_x = sp.symbols('C_x')
k = sp.symbols('k')
C_TF = sp.symbols('C_TF')
r1, r2 = sp.symbols('r1 r2')
rho1 = sp.Function('rho')(r1)
rho2 = sp.Function('rho')(r2)
delta_rho1 = sp.Function('delta_rho')(r1)
delta_rho2 = sp.Function('delta_rho')(r2)

# Example 1: XC Energy Functional
integrand_xc = C_x * rho**(4/3)
functional_derivative_xc = compute_functional_derivative_taylor(None, integrand_xc, rho, delta_rho)
print("Functional Derivative (XC Energy):")
print(functional_derivative_xc)

# Example 2: Hartree Energy Functional (using new double integral method)
integrand_hartree = (rho1 * rho2) / sp.Abs(r1 - r2)
functional_derivative_hartree = compute_double_integral_derivative(
    integrand_hartree,
    r1, r2,
    rho1, rho2,
    delta_rho1, delta_rho2
)
print("\nFunctional Derivative (Hartree Energy):")
print(functional_derivative_hartree)

# Example 3: Thomas–Fermi Kinetic Energy Functional
integrand_tf = C_TF * rho**(5/3)
functional_derivative_tf = compute_functional_derivative_taylor(None, integrand_tf, rho, delta_rho)
print("\nFunctional Derivative (Thomas–Fermi Kinetic Energy):")
print(functional_derivative_tf)

# Example 4: Generic double integral example
def compute_generic_double_integral_derivative(integrand_expr, vars_tuple, funcs_tuple, deltas_tuple):
    """
    Generic method for any double integral functional derivative.
    
    Args:
        integrand_expr: The integrand expression
        vars_tuple: Tuple of integration variables (var1, var2)
        funcs_tuple: Tuple of functions (func1, func2)
        deltas_tuple: Tuple of perturbations (delta1, delta2)
    
    Returns:
        The functional derivative
    """
    return compute_double_integral_derivative(
        integrand_expr,
        vars_tuple[0], vars_tuple[1],
        funcs_tuple[0], funcs_tuple[1],
        deltas_tuple[0], deltas_tuple[1]
    )

# Example usage for a generic double integral
# For any integrand of the form f(x,y) * g(x,y)
x, y = sp.symbols('x y')
f = sp.Function('f')(x)
g = sp.Function('g')(y)
delta_f = sp.Function('delta_f')(x)
delta_g = sp.Function('delta_g')(y)

generic_integrand = f * g / (x - y)**2  # Example generic integrand
generic_derivative = compute_generic_double_integral_derivative(
    generic_integrand,
    (x, y),
    (f, g),
    (delta_f, delta_g)
)
print("\nGeneric Double Integral Derivative Example:")
print(generic_derivative)