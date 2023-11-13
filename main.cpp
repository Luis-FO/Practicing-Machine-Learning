#include <iostream>
#include "neuron.cpp"
#include 
using namespace std;


int main(){
    neuron n = neuron();
    cout<<n.getBias()<<endl;
    cout<<n.getWeight()<<endl;
    
    return 0;
}