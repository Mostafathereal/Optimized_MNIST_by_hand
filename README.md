# Optimized_MNIST_by_hand

## Optimization Algorithms

### Momentum Gradient Descent

Plain gradient descent limits you from increasing the learning rate to avoid noise near the minimum, and divergence. Momentum reduces this noise and allows you to more safely apporach the min point, meaning you can safely increase the learning rate (to an extent). Momentum Gradient Descent almost always works faster than plain Gradient Descent

- Compute regular gradients
- Compute exponentially weighted averages of gradients
- update the weights and biases with these new gradients

Momentum Gradient Descent:

<a href="https://www.codecogs.com/eqnedit.php?latex=\\On&space;\&space;Iteration&space;\&space;t;&space;\\&space;\indent{}&space;Compute&space;\&space;\partial&space;w&space;\&space;and&space;\&space;\partial&space;b&space;\\&space;\indent{}&space;V&space;\partial&space;w&space;=&space;\beta&space;V&space;\partial&space;w&space;&plus;&space;(1-\beta)\partial&space;w&space;\\&space;\indent{}&space;V&space;\partial&space;b&space;=&space;\beta&space;V&space;\partial&space;b&space;&plus;&space;(1-\beta)\partial&space;b&space;\\&space;\indent{}&space;w&space;:=&space;w&space;-&space;\alpha&space;V&space;\partial&space;w&space;\\&space;\indent{}&space;b&space;:=&space;b&space;-&space;\alpha&space;V&space;\partial&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\On&space;\&space;Iteration&space;\&space;t;&space;\\&space;\indent{}&space;Compute&space;\&space;\partial&space;w&space;\&space;and&space;\&space;\partial&space;b&space;\\&space;\indent{}&space;V&space;\partial&space;w&space;=&space;\beta&space;V&space;\partial&space;w&space;&plus;&space;(1-\beta)\partial&space;w&space;\\&space;\indent{}&space;V&space;\partial&space;b&space;=&space;\beta&space;V&space;\partial&space;b&space;&plus;&space;(1-\beta)\partial&space;b&space;\\&space;\indent{}&space;w&space;:=&space;w&space;-&space;\alpha&space;V&space;\partial&space;w&space;\\&space;\indent{}&space;b&space;:=&space;b&space;-&space;\alpha&space;V&space;\partial&space;b" title="\\On \ Iteration \ t; \\ \indent{} Compute \ \partial w \ and \ \partial b \\ \indent{} V \partial w = \beta V \partial w + (1-\beta)\partial w \\ \indent{} V \partial b = \beta V \partial b + (1-\beta)\partial b \\ \indent{} w := w - \alpha V \partial w \\ \indent{} b := b - \alpha V \partial b" /></a>

### RMSProp - Root Mean Squared Prop.A
Again, this allows us to increase the learning rate by dampening the oscillations (noise) as we move closer to the minimum (of the cost func.)

RMSProp:
<a href="https://www.codecogs.com/eqnedit.php?latex=\\On&space;\&space;Iteration&space;\&space;t;&space;\\&space;\indent{}&space;Compute&space;\&space;\partial&space;w&space;\&space;and&space;\&space;\partial&space;b&space;\\&space;\indent{}&space;S&space;\partial&space;w&space;=&space;\beta&space;_2S&space;\partial&space;w&space;&plus;&space;(1-\beta_2)\partial&space;w^2&space;\\&space;\indent{}&space;S&space;\partial&space;b&space;=&space;\beta&space;_2S&space;\partial&space;b&space;&plus;&space;(1-\beta_2)\partial&space;b^2&space;\\&space;\indent{}&space;w&space;:=&space;w&space;-&space;\alpha&space;\frac{\partial&space;w}{\sqrt{S&space;\partial&space;w}&space;&plus;&space;\varepsilon&space;}&space;\\&space;\indent{}&space;b&space;:=&space;b&space;-&space;\alpha&space;\frac{\partial&space;b}{\sqrt{S&space;\partial&space;b}&space;&plus;&space;\varepsilon&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\On&space;\&space;Iteration&space;\&space;t;&space;\\&space;\indent{}&space;Compute&space;\&space;\partial&space;w&space;\&space;and&space;\&space;\partial&space;b&space;\\&space;\indent{}&space;S&space;\partial&space;w&space;=&space;\beta&space;_2S&space;\partial&space;w&space;&plus;&space;(1-\beta_2)\partial&space;w^2&space;\\&space;\indent{}&space;S&space;\partial&space;b&space;=&space;\beta&space;_2S&space;\partial&space;b&space;&plus;&space;(1-\beta_2)\partial&space;b^2&space;\\&space;\indent{}&space;w&space;:=&space;w&space;-&space;\alpha&space;\frac{\partial&space;w}{\sqrt{S&space;\partial&space;w}&space;&plus;&space;\varepsilon&space;}&space;\\&space;\indent{}&space;b&space;:=&space;b&space;-&space;\alpha&space;\frac{\partial&space;b}{\sqrt{S&space;\partial&space;b}&space;&plus;&space;\varepsilon&space;}" title="\\On \ Iteration \ t; \\ \indent{} Compute \ \partial w \ and \ \partial b \\ \indent{} S \partial w = \beta _2S \partial w + (1-\beta_2)\partial w^2 \\ \indent{} S \partial b = \beta _2S \partial b + (1-\beta_2)\partial b^2 \\ \indent{} w := w - \alpha \frac{\partial w}{\sqrt{S \partial w} + \varepsilon } \\ \indent{} b := b - \alpha \frac{\partial b}{\sqrt{S \partial b} + \varepsilon }" /></a>

### Adam Optimization
This algorithm puts Momentum and RMSProp together. It is shown to generalize well and work over a wide range of applications. Note that the typical implementation of Adam Optimization includes bias correction where as Momentum and RMSProp dont. Bias correction is
