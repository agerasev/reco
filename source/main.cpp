#include <iostream>
#include <string>

#include <nn/bp/net.hpp>

#include <nn/sw/bp/layerext.hpp>
#include <nn/sw/bp/conn.hpp>

#include <nn/hw/bp/factory.hpp>
#include <nn/hw/bp/layer.hpp>
#include <nn/hw/bp/conn.hpp>

#include "reader.hpp"
#include "print.hpp"
#include <iostream>
#include <string>
#include <nn/bp/net.hpp>
#include <nn/sw/bp/layerext.hpp>
#include <nn/sw/bp/conn.hpp>

#include "reader.hpp"

int main(int argc, char *argv[])
{
	static const int LAYER_COUNT = 3;
	const int LAYER_SIZE[LAYER_COUNT] = {28*28, 30, 10};
	
	srand(987654);
	
	FactoryHW_BP factory("libnn/opencl/kernel.c");
	
	Net_BP net;
	
	Layer_BP *in;
	Layer_BP *out;
	
	for(int i = 0; i < LAYER_COUNT; ++i)
	{
		Layer_BP *layer;
		if(i != 0)
			layer = factory.newLayer(i, LAYER_SIZE[i], LayerFunc::SIGMOID);
		else
			layer = factory.newLayer(i, LAYER_SIZE[i]);
		
		if(i == 0)
			in = layer;
		else if(i == LAYER_COUNT - 1)
			out = layer;
		net.addLayer(layer);
	}
	
	for(int i = 0; i < LAYER_COUNT - 1; ++i)
	{
		Conn_BP *conn = factory.newConn(i, LAYER_SIZE[i], LAYER_SIZE[i + 1]);
		conn->getWeight().randomize();
		conn->getBias().randomize();
		net.addConn(conn, i, i + 1);
	}
	
	ImageSet train_set("mnist/train-labels.idx1-ubyte", "mnist/train-images.idx3-ubyte");
	ImageSet test_set("mnist/t10k-labels.idx1-ubyte", "mnist/t10k-images.idx3-ubyte");
	
	if(train_set.getImageSizeX() != 28 || train_set.getImageSizeY() != 28)
	{
		std::cerr << "train set image size is not 28x28" << std::endl;
		return 1;
	}
	if(test_set.getImageSizeX() != 28 || test_set.getImageSizeY() != 28)
	{
		std::cerr << "test set image size is not 28x28" << std::endl;
		return 1;
	}
	
	const int batch_size = 10;
	
	float cost;
	int score;
	
	for(int k = 0; k < 0x10; ++k)
	{
		std::cout << "epoch " << k << ':' << std::endl;
		
		score = 0;
		cost = 0.0f;
		for(int j = 0; j < train_set.getSize(); ++j)
		{
			const int out_size = LAYER_SIZE[LAYER_COUNT - 1];
			
			const float *in_data = train_set.getImages()[j].getData().data();
			float out_data[out_size];
			float result[out_size];
			
			int digit = train_set.getImages()[j].getDigit();
			for(int i = 0; i < out_size; ++i)
			{
				result[i] = i == digit ? 1.0f : 0.0f;
			}
			
			in->getInput().write(in_data);
			
			for(int i = 0; i < LAYER_COUNT; ++i)
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
			
			cost += out->getCost(result);
			out->setDesiredOutput(result);
			
			for(int i = 0; i < LAYER_COUNT - 1; ++i)
			{
				net.stepBackward();
			}
			
			if((j + 1) % batch_size == 0)
			{
				net.commitGrad(1.0f);
			}
		}
		
		std::cout << "train set:" << std::endl;
		std::cout << "score: " << score << " / " << train_set.getSize() << std::endl;
		std::cout << "average cost: " << cost/train_set.getSize() << std::endl;
		
		score = 0;
		for(int j = 0; j < test_set.getSize(); ++j)
		{
			const int out_size = LAYER_SIZE[LAYER_COUNT - 1];
			
			const float *in_data = test_set.getImages()[j].getData().data();
			float out_data[out_size];
			int digit = test_set.getImages()[j].getDigit();
			
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
		std::cout << "score: " << score << " / " << test_set.getSize() << std::endl;
		
		std::cout << std::endl;
	}
	
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

