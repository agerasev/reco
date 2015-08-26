#include <iostream>
#include <string>
#include <nn/bp/net.hpp>
#include <nn/sw/bp/layerext.hpp>
#include <nn/sw/bp/conn.hpp>

#include "reader.hpp"

std::ostream &operator << (std::ostream &os, const Buffer &buffer)
{
	int size = buffer.getSize();
	float *data = new float[size];
	buffer.read(data);
	// os << buffer.isValid() << ":  ";
	for(int i = 0; i < size; ++i)
		os << data[i] << ' ';
	delete[] data;
	return os;
}

void printLayer(const Layer *layer)
{
	std::cout << layer->getInput() << std::endl;
	std::cout << layer->getOutput() << std::endl;
}

void printConn(const Conn *conn)
{
	std::cout << conn->getWeight() << std::endl;
	std::cout << conn->getBias() << std::endl;
}

void printLayerError(const Layer_BP *layer)
{
	std::cout << layer->getInputError() << std::endl;
	std::cout << layer->getOutputError() << std::endl;
}

void printConnGrad(const Conn_BP *conn)
{
	std::cout << conn->getWeightGrad() << std::endl;
	std::cout << conn->getBiasGrad() << std::endl;
}

//#define PRINT_DEBUG

int main(int argc, char *argv[])
{
	static const int LAYER_COUNT = 3;
	const int LAYER_SIZE[LAYER_COUNT] = {28*28, 30, 10};
	
	srand(987654);
	
	Net_BP net;
	
	Layer_BP *in;
	Layer_BP *out;
	
	for(int i = 0; i < LAYER_COUNT; ++i)
	{
		Layer_BP *layer;
		if(i != 0)
			layer = new LayerExtSW_BP<LayerFunc::SIGMOID>(i, LAYER_SIZE[i]);
		else
			layer = new LayerSW_BP(i, LAYER_SIZE[i]);
		
		if(i == 0)
			in = layer;
		else if(i == LAYER_COUNT - 1)
			out = layer;
		net.addLayer(layer);
	}
	
	for(int i = 0; i < LAYER_COUNT - 1; ++i)
	{
		Conn_BP *conn = new ConnSW_BP(i, LAYER_SIZE[i], LAYER_SIZE[i + 1]);
		conn->getWeight().randomize();
		conn->getBias().randomize();
		net.addConn(conn, i, i + 1);
	}
	
	ImageSet *train_set = createImageSet("mnist/train-labels.idx1-ubyte", "mnist/train-images.idx3-ubyte");
	ImageSet *test_set = createImageSet("mnist/t10k-labels.idx1-ubyte", "mnist/t10k-images.idx3-ubyte");
	
	if(train_set == nullptr)
	{
		std::cerr << "train set file error" << std::endl;
		return 2;
	}
	if(test_set == nullptr)
	{
		std::cerr << "train set file error" << std::endl;
		return 2;
	}
	
	if(train_set->size_x != 28 || train_set->size_y != 28)
	{
		std::cerr << "train set image size is not 28x28" << std::endl;
		return 1;
	}
	if(test_set->size_x != 28 || test_set->size_y != 28)
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
		for(int j = 0; j < train_set->size; ++j)
		{
			const int out_size = LAYER_SIZE[LAYER_COUNT - 1];
			
			float *in_data = train_set->images[j]->data;
			float out_data[out_size];
			float result[out_size];
			
			int digit = train_set->images[j]->digit;
			for(int i = 0; i < out_size; ++i)
			{
				result[i] = i == digit ? 1.0f : 0.0f;
			}
			
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
			
			cost += out->getCost(result);
			out->setDesiredOutput(result);
			
			for(int i = 0; i < LAYER_COUNT; ++i)
			{
				if(i != 0)
				{
					net.stepBackward();
				}
				
#ifdef PRINT_DEBUG
				//printLayer(net.getLayer(0));
				//printConn(net.getConn(0));
				//printLayer(net.getLayer(1));
				//printConn(net.getConn(1));
				//printLayer(net.getLayer(2));
				
				//printLayerError(dynamic_cast<const Layer_BP *>(net.getLayer(0)));
				//printConnGrad(dynamic_cast<const Conn_BP *>(net.getConn(0)));
				printLayerError(dynamic_cast<const Layer_BP *>(net.getLayer(1)));
				//printConnGrad(dynamic_cast<const Conn_BP *>(net.getConn(1)));
				printLayerError(dynamic_cast<const Layer_BP *>(net.getLayer(2)));
				
				std::cout << std::endl;
#endif
			}
			
			if((j + 1) % batch_size == 0)
			{
				net.commitGrad(1.0f);
			}
		}
		
		std::cout << "train set:" << std::endl;
		std::cout << "score: " << score << " / " << train_set->size << std::endl;
		std::cout << "average cost: " << cost/train_set->size << std::endl;
		
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
	}
	
	destroyImageSet(train_set);
	destroyImageSet(test_set);
	
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
