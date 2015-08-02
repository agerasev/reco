#include <iostream>
#include <nn/network.hpp>
#include <nn/opencl/factory.hpp>

int main(int argc, char *argv[])
{
	nn::Network net;
	nn::cl::Factory factory("libnn/opencl/kernel.c");
	
	nn::cl::Layer *in = factory.createLayer(1,4);
	nn::cl::Layer *out = factory.createLayer(2,4);
	nn::cl::Connection *conn = factory.createConnection(1,4,4);
	
	net.addLayer(in);
	net.addLayer(out);
	net.addConnection(conn, in->getID(), out->getID());
	
	float in_data[4] = {1,2,3,4};
	float weight_data[16] = {1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1};
	float bias_data[4] = {1,1,1,1};
	
	in->write(in_data);
	conn->write_weight(weight_data);
	conn->write_bias(bias_data);
	
	for(int i = 0; i < 2; ++i)
		net.stepForward();
	
	float out_data[4];
	
	out->read(out_data);
	
	for(int i = 0; i < 4; ++i)
		std::cout << out_data[i] << ' ';
	std::cout << std::endl;
	
	return 0;
}
