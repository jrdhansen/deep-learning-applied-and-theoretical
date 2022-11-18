




library(matrixcalc)  # This library has the hadamard.prod function
library(e1071)   # This library has the sigmoid function, and first derivative
# of the sigmoid function (dsigmoid)

#==== FORWARD PASS =============================================================

w1 = matrix(c(0.15, 0.25, 0.20, 0.30), byrow = TRUE, nrow = 2)
i = matrix(c(0.05, 0.1), byrow = TRUE, nrow = 2)
b1 = matrix(c(0.35, 0.35), byrow = TRUE, nrow = 2)
z_hid = (w1 %*% i) + b1
a_hid = sigmoid(z_hid)

w2 = matrix(c(0.40, 0.50, 0.45, 0.55), byrow = TRUE, nrow = 2)
b2 = matrix(c(0.6, 0.6), byrow = TRUE, nrow = 2)
z_out = (w2 %*% a_hid) + b2
a_out = sigmoid(z_out)

y = matrix(c(0.01, 0.99), byrow = TRUE, nrow = 2)

grad_C = matrix(c(-y[1]/a_out[1] + (1-y[1])/(1-a_out[1]),
                  -y[2]/a_out[2] + (1-y[2])/(1-a_out[2])))

sig_prime_z_out = matrix(dsigmoid(z_out))

# Compute the errors of the output layer ===== THIS IS ONE OF THE ANSWERS
#===============================================================================
delta_out = hadamard.prod(grad_C, sig_prime_z_out)
View(delta_out)

# By defn, delta_hid = [t(w^out)][delta_out] hadamard [sig_prime_z_hid]
# where t(w^out) is the transpose of w^2
sig_prime_z_hid = matrix(dsigmoid(z_hid))

# Compute the errors of the hidden layer ===== THIS IS ONE OF THE ANSWERS
#===============================================================================
delta_hid = hadamard.prod(t(w2)%*%delta_out, sig_prime_z_hid)
View(delta_hid)



# By the 4th Fundamental Equation of Backpropogation, we know that:
# (partial C/partial w_{j,k}^l) = a_{k}^{l-1} * delta_j^l
# In other words, the partial wrt to some weight is equal to the activation
# in the previous layer * the error in the next layer (where the weight is
# connecting the activation neuron in {l-1} and the delta neuron in {l}).
#==== THESE ARE ALL ANSWERS ====================================================
partial_w1 = i[1] * delta_hid[1]
partial_w2 = i[1] * delta_hid[2]
partial_w3 = i[2] * delta_hid[1]
partial_w4 = i[2] * delta_hid[2]
partial_w5 = a_hid[1] * delta_out[1]
partial_w6 = a_hid[1] * delta_out[2]
partial_w7 = a_hid[2] * delta_out[1]
partial_w8 = a_hid[2] * delta_out[2]

partials_wrt_weights = c(partial_w1,
                         partial_w2,
                         partial_w3,
                         partial_w4,
                         partial_w5,
                         partial_w6,
                         partial_w7,
                         partial_w8)
View(partials_wrt_weights)

# By the 3rd Fundamental Equation of Backpropogation, we know that:
# (partial C/partial b_j^l) = delta_j^l
# Per our notation, this means that 
#==== THESE ARE ALL ANSWERS ====================================================
partial_b11 = delta_hid[1]
partial_b12 = delta_hid[2]
partial_b21 = delta_out[1]
partial_b22 = delta_out[2]

partials_wrt_biases = c(partial_b11,
                        partial_b12,
                        partial_b21,
                        partial_b22)
View(partials_wrt_biases)

