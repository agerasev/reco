#include <iostream>
#include <string>
#include <random>
#include <nn/bp/net.hpp>
#include <nn/sw/bp/layerext.hpp>
#include <nn/sw/bp/conn.hpp>

float urandom01()
{
	return float(random())/RAND_MAX;
}

float urandom11()
{
	return 2.0f*urandom01() - 1.0f;
}

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

void printStep(int i, const Layer_BP *in, const Layer_BP *out)
{
	std::cout << "step " << i << ':' << std::endl;
	std::cout << in->getInput() << '\t' << in->getOutputError() << std::endl;
	std::cout << in->getOutput() << '\t' << in->getInputError() << std::endl;
	std::cout << out->getInput() << '\t' << out->getOutputError() << std::endl;
	std::cout << out->getOutput() << '\t' << out->getInputError() << std::endl;
}

void printConn(const Conn_BP *conn)
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

int main(int argc, char *argv[])
{
	srand(12345);
	
	Net_BP net;
	
	const Layer::ID in_id = 1, out_id = 2;
	const Conn::ID conn_id = 1;
	static const int in_size = 2, out_size = 4;
	
	Layer_BP *in;
	Layer_BP *out;
	Conn_BP *conn;
	
	in = new LayerSW_BP(in_id, in_size);
	out = new LayerSW_BP(out_id, out_size);
	conn = new ConnSW_BP(conn_id, in_size, out_size);
	
	net.addLayer(in);
	net.addLayer(out);
	net.addConn(conn, in->getID(), out->getID());
	
	float weight_data[in_size*out_size];
	for(int i = 0; i < in_size*out_size; ++i)
	{
		weight_data[i] = urandom11();
	}
	float bias_data[out_size];
	for(int i = 0; i < out_size; ++i)
	{
		bias_data[i] = urandom11();
	}
	
	conn->getWeight().write(weight_data);
	conn->getBias().write(bias_data);
	
	for(int j = 0; j < 10000; ++j)
	{
		int num = random() % 4;
		float in_data[in_size];
		float out_data[out_size];
		float result[out_size];
		for(int i = 0; i < in_size; ++i)
		{
			in_data[i] = (num >> i) & 1;
		}
		for(int i = 0; i < out_size; ++i)
		{
			result[i] = i == num ? 1.0f : 0.0f;
		}
		std::cout << num << std::endl;
		
		in->getInput().write(in_data);
		
		for(int i = 0; i < 3; ++i)
		{
			//printStep(i, in, out);
			net.stepForward();
		}
		
		out->getOutput().read(out_data);
		for(int i = 0; i < in_size; ++i)
		{
			result[i] -= out_data[i];
		}
		out->getInputError().write(result);
		out->getInputError().validate(true);
		
		for(int i = 0; i < 2; ++i)
		{
			//printStep(i, in, out);
			//printConn(conn);
			net.stepBackward();
		}
		
		if(j % 10 == 0)
		{
			printStep(j, in, out);
			printConn(conn);
			net.commitGrad(1e-2);
		}
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
