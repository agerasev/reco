#include <iostream>
#include <string>
#include <nn/bp/net.hpp>
#include <nn/sw/bp/layerext.hpp>
#include <nn/sw/bp/conn.hpp>

#include "reader.hpp"

std::ostream &operator << (std::ostream &os, const Layer::Buffer &buffer)
{
	int size = buffer.getSize();
	float *data = new float[size];
	buffer.read(data);
	os << buffer.isValid() << ":  ";
	for(int i = 0; i < size; ++i)
		os << data[i] << ' ';
	delete[] data;
	return os;
}

void printLayer_BP(const Layer_BP *layer)
{
	std::cout << layer->getInput() << '\t' << layer->getOutputError() << std::endl;
	std::cout << layer->getOutput() << '\t' << layer->getInputError() << std::endl;
}

void printConn_BP(const Conn_BP *conn)
{
	int sx = conn->getInputSize(), sy = conn->getOutputSize();
	float *data = new float[2*(sx + 1)*sy];
	conn->getWeight().read(data);
	conn->getBias().read(data + sx*sy);
	conn->getWeightGrad().read(data + (sx + 1)*sy);
	conn->getBiasGrad().read(data + (sx + 1)*sy + sx*sy);
	for(int iy = 0; iy < sy; ++iy)
	{
		for(int ix = 0; ix < sx; ++ix)
		{
			std::cout << data[iy*sx + ix] << ' ';
		}
		std::cout << ' ' << data[sx*sy + iy] << '\t';
		for(int ix = 0; ix < sx; ++ix)
		{
			std::cout << data[(sx + 1)*sy + iy*sx + ix] << ' ';
		}
		std::cout << ' ' << data[(sx + 1)*sy + sx*sy + iy] << std::endl;
	}
	delete[] data;
}

#define PRINT_COST
//#define PRINT_FORWARD
//#define PRINT_BACKWARD

int main(int argc, char *argv[])
{
	static const int LAYER_COUNT = 3;
	const int LAYER_SIZE[LAYER_COUNT] = {28*28, 30, 10};
	
	srand(32526);
	
	Net_BP net;
	
	Layer_BP *in;
	Layer_BP *out;
	
	for(int i = 0; i < LAYER_COUNT; ++i)
	{
		Layer_BP *layer = new LayerExtSW_BP<EXT_SIGMOID>(i, LAYER_SIZE[i]);
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
	
	const int epoch_length = train_set->size;
	const int batch_size = 10;
	
#ifdef PRINT_COST
	const int cost_count = 10;
	float cost = 0.0;
#endif // PRINT_COST
	for(int j = 0; j < epoch_length; ++j)
	{
		// const int in_size = LAYER_SIZE[0];
		const int out_size = LAYER_SIZE[LAYER_COUNT - 1];
		
		float *in_data = train_set->images[j]->data;
		float out_data[out_size];
		float result[out_size];
		for(int i = 0; i < out_size; ++i)
		{
			result[i] = i == train_set->images[j]->digit ? 1.0f : 0.0f;
		}
		
		in->getInput().write(in_data);
		
		for(int i = 0; i < LAYER_COUNT + 1; ++i)
		{
			if(i != 0)
				net.stepForward();
#ifdef PRINT_FORWARD
			net.forLayers([](Layer *l)
			{
				Layer_BP *lb = dynamic_cast<Layer_BP *>(l);
				if(lb != nullptr)
					printLayer_BP(lb);
			});
			std::cout << std::endl;
#endif // PRINT_FORWARD
		}
		
		
		out->getOutput().read(out_data);
		for(int i = 0; i < out_size; ++i)
		{
			result[i] -= out_data[i];
		}
		out->getInputError().write(result);
		out->getInputError().validate(true);
		
#ifdef PRINT_COST
		cost += out->getCost();
#endif // PRINT_COST
		
		for(int i = 0; i < LAYER_COUNT; ++i)
		{
			if(i != 0)
				net.stepBackward();
#ifdef PRINT_BACKWARD
			net.forLayers([](Layer *l)
			{
				Layer_BP *lb = dynamic_cast<Layer_BP *>(l);
				if(lb != nullptr)
					printLayer_BP(lb);
			});
			net.forConns([](Conn *c)
			{
				Conn_BP *cb = dynamic_cast<Conn_BP *>(c);
				if(cb != nullptr)
					printConn_BP(cb);
			});
			std::cout << std::endl;
#endif // PRINT_BACKWARD
		}
		
		if((j + 1) % batch_size == 0)
		{
			net.commitGrad(1.0);
		}
		
#ifdef PRINT_COST
		if((j + 1) % (epoch_length/cost_count) == 0)
		{
			std::cout << "average cost: " << cost/(epoch_length/cost_count) << std::endl;
			cost = 0.0f;
		}
#endif // PRINT_COST
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
