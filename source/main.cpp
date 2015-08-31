#include <iostream>
#include <string>

#include <nn/net.hpp>

#include <nn/sw/layerext.hpp>
#include <nn/sw/conn.hpp>

#include <nn/hw/factory.hpp>
#include <nn/hw/layerext.hpp>
#include <nn/hw/conn.hpp>

#include "reader.hpp"
#include "print.hpp"

int main(int argc, char *argv[])
{
	const int LAYER_SIZE[3] = {28*28, 30, 10};
	
	unsigned seed = 987654;
	
	FactoryHW factory("libnn/opencl/kernel.c");
	
	Net net_sw, net_hw;
	
	Layer *in_sw, *in_hw;
	Layer *out_sw, *out_hw;
	
	for(int i = 0; i < 3; ++i)
	{
		LayerSW *layer_sw;
		LayerHW *layer_hw;
		if(i != 0)
		{
			layer_sw = new LayerExtSW<LayerFunc::SIGMOID>(i, LAYER_SIZE[i]);
			layer_hw = factory.newLayer(i, LAYER_SIZE[i], LayerFunc::SIGMOID);
		}
		else
		{
			layer_sw = new LayerSW(i, LAYER_SIZE[i]);
			layer_hw = factory.newLayer(i, LAYER_SIZE[i]);
		}
		
		if(i == 0)
		{
			in_sw = layer_sw;
			in_hw = layer_hw;
		}
		else if(i == 2)
		{
			out_sw = layer_sw;
			out_hw = layer_hw;
		}
		net_sw.addLayer(layer_sw);
		net_hw.addLayer(layer_hw);
	}
	
	srand(seed);
	for(int i = 0; i < 2; ++i)
	{
		Conn *conn = new ConnSW(i, LAYER_SIZE[i], LAYER_SIZE[i + 1]);
		conn->getWeight().randomize();
		conn->getBias().randomize();
		net_sw.addConn(conn, i, i + 1);
	}
	
	srand(seed);
	for(int i = 0; i < 2; ++i)
	{
		Conn *conn = factory.newConn(i, LAYER_SIZE[i], LAYER_SIZE[i + 1]);
		conn->getWeight().randomize();
		conn->getBias().randomize();
		net_hw.addConn(conn, i, i + 1);
	}
	
	ImageSet test_set("mnist/t10k-labels.idx1-ubyte", "mnist/t10k-images.idx3-ubyte");
	
	if(test_set.getImageSizeX() != 28 || test_set.getImageSizeY() != 28)
	{
		std::cerr << "test set image size is not 28x28" << std::endl;
		return 1;
	}
	
	for(int j = 0; j < test_set.getSize(); ++j)
	{
		const float *in_data = test_set.getImages()[j].getData().data();
		float out_data_sw[10], out_data_hw[10];
		
		in_sw->getInput().write(in_data);
		in_hw->getInput().write(in_data);
		
		for(int i = 0; i < 3; ++i)
		{
			net_sw.stepForward();
			net_hw.stepForward();
		}
		
		out_sw->getOutput().read(out_data_sw);
		out_hw->getOutput().read(out_data_hw);
		
		float diff = 0.0f;
		for(int i = 0; i < 10; ++i)
		{
			diff += out_data_sw[i] - out_data_hw[i];
		}
		std::cout << j << " difference: " << diff << std::endl;
	}
	
	net_sw.forConns([](Conn *conn)
	{
		delete conn;
	});
	net_sw.forLayers([](Layer *layer)
	{
		delete layer;
	});
	
	net_hw.forConns([](Conn *conn)
	{
		delete conn;
	});
	net_hw.forLayers([](Layer *layer)
	{
		delete layer;
	});
	
	return 0;
}
