#include <iostream>
#include <string>

#include <nn/net.hpp>

#include <nn/sw/layerext.hpp>
#include <nn/sw/conn.hpp>

#include <nn/hw/factory.hpp>
#include <nn/hw/layerext.hpp>
#include <nn/hw/conn.hpp>

#include "reader.hpp"

int main(int argc, char *argv[])
{
	static const int LAYER_COUNT = 3;
	const int LAYER_SIZE[LAYER_COUNT] = {28*28, 30, 10};
	
	unsigned seed = 987654;
	
	FactoryHW factory("libnn/opencl/kernel.c");
	
	Net net_sw, net_hw;
	
	Layer *in_sw, *in_hw;
	Layer *out_sw, *out_hw;
	
	for(int i = 0; i < LAYER_COUNT; ++i)
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
		else if(i == LAYER_COUNT - 1)
		{
			out_sw = layer_sw;
			out_hw = layer_hw;
		}
		net_sw.addLayer(layer_sw);
		net_hw.addLayer(layer_hw);
	}
	
	srand(seed);
	for(int i = 0; i < LAYER_COUNT - 1; ++i)
	{
		Conn *conn = new ConnSW(i, LAYER_SIZE[i], LAYER_SIZE[i + 1]);
		conn->getWeight().randomize();
		conn->getBias().randomize();
		net_sw.addConn(conn, i, i + 1);
	}
	
	srand(seed);
	for(int i = 0; i < LAYER_COUNT - 1; ++i)
	{
		Conn *conn = factory.newConn(i, LAYER_SIZE[i], LAYER_SIZE[i + 1]);
		conn->getWeight().randomize();
		conn->getBias().randomize();
		net_hw.addConn(conn, i, i + 1);
	}
	
	ImageSet *test_set = createImageSet("mnist/t10k-labels.idx1-ubyte", "mnist/t10k-images.idx3-ubyte");
	
	if(test_set == nullptr)
	{
		std::cerr << "train set file error" << std::endl;
		return 2;
	}
	
	if(test_set->size_x != 28 || test_set->size_y != 28)
	{
		std::cerr << "test set image size is not 28x28" << std::endl;
		return 1;
	}
	
	for(int i = 0; i < LAYER_COUNT - 1; ++i)
	{
		if(
		   net_sw.getConn(i)->getWeight().getSize() != 
		   net_hw.getConn(i)->getWeight().getSize() ||
		   net_sw.getConn(i)->getBias().getSize() != 
		   net_hw.getConn(i)->getBias().getSize()
		   )
		{
			std::cerr << "weight and bias sizes not match" << std::endl;
			return 3;
		}
		
		int weight_size = net_sw.getConn(i)->getWeight().getSize();
		int bias_size = net_sw.getConn(i)->getBias().getSize();
		
		float *weight_sw, *weight_hw, *bias_sw, *bias_hw;
		weight_sw = new float[weight_size];
		weight_hw = new float[weight_size];
		bias_sw = new float[bias_size];
		bias_hw = new float[bias_size];
		
		net_sw.getConn(i)->getWeight().read(weight_sw);
		net_hw.getConn(i)->getWeight().read(weight_hw);
		net_sw.getConn(i)->getBias().read(bias_sw);
		net_hw.getConn(i)->getBias().read(bias_hw);
		
		float weight_diff = 0.0f, bias_diff = 0.0f;
		for(int j = 0; j < weight_size; ++j)
		{
			weight_diff += weight_sw[j] - weight_hw[j];
		}
		for(int j = 0; j < bias_size; ++j)
		{
			bias_diff += bias_sw[j] - bias_hw[j];
		}
		delete[] weight_hw;
		delete[] weight_sw;
		delete[] bias_hw;
		delete[] bias_sw;
		
		std::cout << "conn " << i << ":" << std::endl;
		std::cout << "weight difference: " << weight_diff << std::endl;
		std::cout << "bias difference: " << bias_diff << std::endl;
	}
	
	/*
	score = 0;
	for(int j = 0; j < test_set->size; ++j)
	{
		const int out_size = LAYER_SIZE[LAYER_COUNT - 1];
		
		float *in_data = test_set->images[j]->data;
		float out_data[out_size];
		int digit = test_set->images[j]->digit;
		
		in->getInput().write(in_data);
		
		for(int i = 0; i < LAYER_COUNT + 1; ++i)
		{
			net.stepForward();
		}
		
		out->getOutput().read(out_data);
		
		float max_val = out_data[0];
		int max_digit = 0;
		for(int i = 1; i < 10; ++i)
		{
			if(out_data[i] > max_val)
			{
				max_val = out_data[i];
				max_digit = i;
			}
		}
		if(max_digit == digit)
			++score;
	}
	
	std::cout << "test set:" << std::endl;
	std::cout << "score: " << score << " / " << test_set->size << std::endl;
	
	std::cout << std::endl;
	*/
	
	destroyImageSet(test_set);
	
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
