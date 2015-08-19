#include <iostream>
#include <string>
#include <nn/net.hpp>
#include <nn/sw/layerext.hpp>
#include <nn/sw/conn.hpp>

int main(int argc, char *argv[])
{
	Net net;
	
	const Layer::ID in_id = 1, out_id = 2;
	const Conn::ID conn_id = 1;
	const int in_size = 4, out_size = 4;
	
	Layer *in;
	Layer *out;
	Conn *conn;
	
	in = new LayerSW(in_id, in_size);
	out = new LayerExtSW<EXT_SIGMOID>(out_id, out_size);
	conn = new ConnSW(conn_id, in_size, out_size);
	
	net.addLayer(in);
	net.addLayer(out);
	net.addConn(conn, in->getID(), out->getID());
	
	float in_data[4] = {1,2,3,4};
	float weight_data[16] = {1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1};
	float bias_data[4] = {1,1,1,1};
	
	in->getInput().write(in_data);
	conn->writeWeight(weight_data);
	conn->writeBias(bias_data);
	
	for(int i = 0; i < 2; ++i)
		net.stepForward();
	
	float out_data[4];
	
	out->getOutput().read(out_data);
	
	for(int i = 0; i < 4; ++i)
		std::cout << out_data[i] << ' ';
	std::cout << std::endl;
	
	net.forConns([](Conn *conn) 
	{
		delete conn;
	});
	net.forLayers([](Layer *layer)
	{
		delete layer;
	});
	
	return 0;
}
