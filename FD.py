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
    # Perturb the variable: rho -> rho + delta_rho
    perturbed_variable = perturbation_variable + perturbation

    # Replace the variable in the integrand
    perturbed_integrand = integrand.subs(perturbation_variable, perturbed_variable)

    # Compute the derivative of the perturbed integrand with respect to the perturbation
    functional_derivative = sp.diff(perturbed_integrand, perturbation).subs(perturbation, 0)

    return functional_derivative


# Symbols
r = sp.symbols('r')  # Spatial variable
r_prime = sp.symbols('r_prime')  # Another spatial variable for the double integral
rho = sp.Function('rho')(r)  # Density function at r
rho_prime = sp.Function('rho')(r_prime)  # Density function at r_prime
delta_rho = sp.Function('delta_rho')(r)  # Perturbation to the density
delta_rho1 = sp.Function('delta_rho')(r_prime)  # Perturbation to rho_prime
C_x = sp.symbols('C_x')  # Constant for XC energy
k = sp.symbols('k')  # Coulomb constant
C_TF = sp.symbols('C_TF') 
r1, r2 = sp.symbols('r1 r2')  # Spatial variables for r1 and r2
rho1 = sp.Function('rho')(r1)  # Density at r1
rho2 = sp.Function('rho')(r2)  # Density at r2
delta_rho1 = sp.Function('delta_rho')(r1)  # Perturbation at r1


# Example 1: XC Energy Functional
# Define the integrand for XC energy functional
integrand_xc = C_x * rho**(4/3)

# Compute the functional derivative using Taylor expansion
functional_derivative_xc = compute_functional_derivative_taylor(None, integrand_xc, rho, delta_rho)
print("Functional Derivative (XC Energy):")
print(functional_derivative_xc)


# Example 2: Hartree Energy Functional
# Define the integrand for Hartree energy functional
integrand_hartree = (rho1 * rho2) / sp.Abs(r1 - r2)

# Compute the functional derivative using Taylor expansion
perturbed_hartree = integrand_hartree.subs(rho1, rho1 + delta_rho1)
functional_derivative_hartree = sp.diff(perturbed_hartree, delta_rho1).subs(delta_rho1, 0)
functional_derivative_hartree = sp.Integral(functional_derivative_hartree, (r2, -sp.oo, sp.oo))
print("\nFunctional Derivative (Hartree Energy):")
print(functional_derivative_hartree)


# Example 3: Thomas–Fermi Kinetic Energy Functional
# Define the integrand for Thomas–Fermi kinetic energy functional
integrand_tf = C_TF * rho**(5/3)

# Compute the functional derivative using Taylor expansion
functional_derivative_tf = compute_functional_derivative_taylor(None, integrand_tf, rho, delta_rho)
print("\nFunctional Derivative (Thomas–Fermi Kinetic Energy):")
print(functional_derivative_tf)
