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

int main(int argc, char *argv[])
{
	const int LAYER_SIZE[3] = {28*28, 30, 10};
	
	unsigned seed = 987654;
	
	FactoryHW_BP factory("libnn/opencl/kernel.c");
	
	Net_BP net_sw, net_hw;
	
	Layer_BP *in_sw, *in_hw;
	Layer_BP *out_sw, *out_hw;
	
	for(int i = 0; i < 3; ++i)
	{
		LayerSW_BP *layer_sw;
		LayerHW_BP *layer_hw;
		if(i != 0)
		{
			layer_sw = new LayerExtSW_BP<LayerFunc::SIGMOID>(i, LAYER_SIZE[i]);
			layer_hw = factory.newLayer(i, LAYER_SIZE[i], LayerFunc::SIGMOID);
		}
		else
		{
			layer_sw = new LayerSW_BP(i, LAYER_SIZE[i]);
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
		Conn *conn = new ConnSW_BP(i, LAYER_SIZE[i], LAYER_SIZE[i + 1]);
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
	
	for(int j = 0; j < 1; ++j)
	{
		const float *in_data = test_set.getImages()[j].getData().data();
		float out_data_sw[10], out_data_hw[10];
		float result[10];
					
		int digit = test_set.getImages()[j].getDigit();
		for(int i = 0; i < 10; ++i)
		{
			result[i] = i == digit ? 1.0f : 0.0f;
		}
		
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
			float val = out_data_sw[i] - out_data_hw[i];
			diff += val*val;
		}
		// std::cout << j << " output difference: " << diff << std::endl;
		
		/*
		float max_val_sw = out_data_sw[0], max_val_hw = out_data_hw[0];
		int max_digit_sw = 0, max_digit_hw = 0;
		for(int i = 1; i < 10; ++i)
		{
			if(out_data_sw[i] > max_val_sw)
			{
				max_val_sw = out_data_sw[i];
				max_digit_sw = i;
			}
			if(out_data_hw[i] > max_val_hw)
			{
				max_val_hw = out_data_hw[i];
				max_digit_hw = i;
			}
		}
		*/
		
		out_sw->setDesiredOutput(result);
		out_hw->setDesiredOutput(result);

		//std::cout << j << " cost difference: " << out_sw->getCost(result) - out_hw->getCost(result) << std::endl;
		
		for(int i = 0; i < 2; ++i)
		{
			net_sw.stepBackward();
			net_hw.stepBackward();
		}

		/*
		int conn_no = 1;
		int weight_grad_size = net_sw.getConn(conn_no)->getWeight().getSize();
		float *weight_grad_sw = new float[weight_grad_size], *weight_grad_hw = new float[weight_grad_size];
		dynamic_cast<const Conn_BP *>(net_sw.getConn(conn_no))->getWeightGrad().read(weight_grad_sw);
		dynamic_cast<const Conn_BP *>(net_hw.getConn(conn_no))->getWeightGrad().read(weight_grad_hw);
		diff = 0.0f;
		for(int i = 0; i < weight_grad_size; ++i)
		{
			float val = weight_grad_sw[i] - weight_grad_hw[i];
			diff += val*val;
		}
		delete[] weight_grad_sw;
		delete[] weight_grad_hw;
		std::cout << j << " weight grad difference: " << diff << std::endl;
		*/

		net_sw.commitGrad(1.0f);
		net_hw.commitGrad(1.0f);

		std::cout << net_sw.getConn(0)->getWeight() << std::endl;
		std::cout << net_hw.getConn(0)->getWeight() << std::endl;
		std::cout << net_sw.getConn(0)->getBias() << std::endl;
		std::cout << net_hw.getConn(0)->getBias() << std::endl;
		std::cout << net_sw.getConn(1)->getWeight() << std::endl;
		std::cout << net_hw.getConn(1)->getWeight() << std::endl;
		std::cout << net_sw.getConn(1)->getBias() << std::endl;
		std::cout << net_hw.getConn(1)->getBias() << std::endl;
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
