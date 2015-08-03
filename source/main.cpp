#include <iostream>
#include <string>
#include <nn/network.hpp>
#include <nn/software/sigmoidlayer.hpp>
#include <nn/software/connection.hpp>
#ifdef NN_OPENCL
#include <nn/opencl/factory.hpp>
#endif // NN_OPENCL


int main(int argc, char *argv[])
{
	nn::Network net;
	
	const nn::Layer::ID in_id = 1, out_id = 2;
	const nn::Connection::ID conn_id = 1;
	const int in_size = 4, out_size = 4;
	
	nn::Layer *in;
	nn::Layer *out;
	nn::Connection *conn;
	
#ifdef NN_OPENCL
	nn::cl::Factory *factory = new nn::cl::Factory("libnn/opencl/kernel.c");
	in = factory->createLayer(in_id, in_size);
	out = factory->createLayer(out_id, out_size);
	conn = factory->createConnection(conn_id, in_size, out_size);
#else // NN_OPENCL
	in = new nn::sw::Layer(in_id, in_size);
	out = new nn::sw::SigmoidLayer(out_id, out_size);
	conn = new nn::sw::Connection(conn_id, in_size, out_size);
#endif // NN_OPENCL
	
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
	
	net.forConnections([](nn::Connection *conn) 
	{
		delete conn;
	});
	net.forLayers([](nn::Layer *layer)
	{
		delete layer;
	});

#ifdef NN_OPENCL
	delete factory;
#endif // NN_OPENCL
	
	return 0;
}
