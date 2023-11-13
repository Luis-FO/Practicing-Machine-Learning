#include <iostream>
#include <cmath>

class neuron
{
    private:
        /* data */
        double bias;
        double weight;
        
    public:
        neuron(/* args */);
        ~neuron();
        double getBias();
        double getWeight();
        double calculate(double x);

};

neuron::neuron(/* args */)
{
    bias = rand();
    weight = rand();
}

neuron::~neuron()
{
}

double neuron::getBias(){
    return bias;
}

double neuron::getWeight(){
    return weight;
}

double neuron::calculate(double x){
    return weight*x + bias;
}

double sigmoid_prime(double z){
    return sigmoid(z) * (1- sigmoid(z));
}

double sigmoid(double z){
    return 1.0/(1.0+exp(-z));
}