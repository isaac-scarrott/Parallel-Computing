#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <string> 

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

#include <iostream>
#include <chrono>

//Creates a data structure to hold data about the weather in vectors and creates a variable using the datastructure created
struct weather_Data {
	vector<string> location;
	vector<int> year;
	vector<int> month;
	vector<int> day;
	vector<int> time;
	vector<int> tempuratureInt;
	vector<float> tempuratureFloat;
};

//Used to make selections about inputs and sends the back by reference - text file (short or long), use float values or integers, workgroup size and if you want to write to a text file
void selection(string &fileName, bool &useFloats, size_t &workgroupSize, bool &writeToFile) {
	int input; //Stores the users input

	//All of these are pretty self explanitory, just read the output
	cout << "\nWhat file do you want to use as an input? 0 for short, 1 for long: ";
	cin >> input;

	switch (input)
	{
	case 0:
		fileName = "short.txt";
		break;
	case 1:
		fileName = "long.txt";
		break;
	default:
		cout << "Please enter either 0 or 1" << endl;
	}

	cout << "\nDo you want to use integers or floats as an input? 0 for integers, 1 for floats: ";
	cin >> input;
	switch (input)
	{
	case 0:
		useFloats = false;
		break;
	case 1:
		useFloats = true;
		break;
	default:
		cout << "Please enter either 0 or 1" << endl;
	}

	cout << "\nWhat workgroup size do you want to use: ";
	cin >> workgroupSize;

	cout << "\nDo you want to write the output to a text file? 0 for no, 1 for yes (might make console display nothing): ";
	cin >> input;
	switch (input)
	{
	case 0:
		writeToFile = false;
		break;
	case 1:
		writeToFile = true;
		break;
	default:
		cout << "Please enter either y or n" << endl;
	}
}

//Function to read the data in and store it in the weather_data structure, will also pass the weather_data structure back
weather_Data readData(string fileName) {
	weather_Data weatherData;
	//Creates an if stream and loads in the text file into infile and a string to tempuarily store each line of code
	ifstream infile(fileName);
	string line;

	//Tokenizes each string into a vector and then loads the vector into the data structure
	while (getline(infile, line)) {
		istringstream iss(line);
		vector<std::string> results((std::istream_iterator<std::string>(iss)),
			std::istream_iterator<std::string>());

		weatherData.location.push_back(results[0]);
		weatherData.year.push_back(stoi(results[1]));
		weatherData.month.push_back(stoi(results[2]));
		weatherData.day.push_back(stoi(results[3]));
		weatherData.time.push_back(stoi(results[4]));
		weatherData.tempuratureInt.push_back(stoi(results[5]));
		weatherData.tempuratureFloat.push_back(stof(results[5]));
	}
	//Closes the stream
	infile.close();

	return weatherData;
}

//Adds padding to the dataset so it can run in parallel
void addPadding(size_t workgroupSize, weather_Data &weatherData) {

	//Finds the padding size needed to be added onto the end of the data
	size_t padding_size = weatherData.tempuratureInt.size() % workgroupSize;
	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected
	if (padding_size) {
		//create an extra vector with neutral values
		vector<int> ext(workgroupSize - padding_size, 0);
		//append that extra vector to our input
		weatherData.tempuratureInt.insert(weatherData.tempuratureInt.end(), ext.begin(), ext.end());
		weatherData.tempuratureFloat.insert(weatherData.tempuratureFloat.end(), ext.begin(), ext.end());
	}
}

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	//0 is GPU, 1 is CPU, 2 is CPU experimental
	int platform_id;
	cout << "Please Choose a platform id: ";
	cin >> platform_id;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels_1.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Used to read in the file name from the user, if they want to use floats or integers and the workgroup size
		string fileName;
		bool useFloats;
		bool writeToFile;
		size_t workgroupSize;

		//Calls the function selection which will allow the user to make choices about which file to use, if they want to use floats, workgroup size and if they want to output the data to a text file
		selection(fileName, useFloats, workgroupSize, writeToFile);

		//Names the file name based on their selections they just made
		if (writeToFile) {
			std::string writeFileName = to_string(platform_id);

			if (useFloats) {
				writeFileName = writeFileName + "-floats-";
			}
			else {
				writeFileName = writeFileName + "-ints-";
			}

			if (fileName == "short.txt") {
				writeFileName = writeFileName + "short-" + to_string(workgroupSize) + ".txt";
			}
			else {
				writeFileName = writeFileName + "long-" + to_string(workgroupSize) + ".txt";
			}
			//This is to write the console to a text file
			freopen(writeFileName.c_str(), "w", stdout);
		}

		//display the selected device
		std::cout << "\nRunning on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//Populates the weatherData variable with the weather data
		weather_Data weatherData = readData(fileName);

		//Calls the function readData to read the data in from the file and store it in a weather_Data structure

		int originalSize = weatherData.tempuratureInt.size(); //The size of the dataset without the padding

		//Calls the function add padding which will add padding depending on the size of the workgroup
		addPadding(workgroupSize, weatherData);

		size_t inputElements = weatherData.tempuratureInt.size();//number of input elements
		size_t inputSize = inputElements * sizeof(int);//size in bytes
		size_t noWorkGroups = inputElements / workgroupSize; //Total number of workgroups

		auto Start = std::chrono::high_resolution_clock::now(); //Starts the clock for the full runtime of the application
		auto bufferInitalisationStart = std::chrono::high_resolution_clock::now(); //Clock for the inalisation of the buffers
		//Creates a buffer to store the output
		cl::Buffer buffer_Output(context, CL_MEM_READ_WRITE, inputSize); //Used to store what is outputted from the kernael on device memory
		queue.enqueueFillBuffer(buffer_Output, 0, 0, inputSize);//zero buffer on device memory
		cl::Buffer buffer_Output2(context, CL_MEM_READ_WRITE, inputSize); //Secondary output buffer for SD
		queue.enqueueFillBuffer(buffer_Output2, 0, 0, inputSize);//zero B buffer on device memory

		//Used to store the execution times of everything
		long long minTime;
		long long maxTime;
		long long meanTime;
		long long SDTime;
		long long medianGlobalTime;
		long long medianLocalTime;
		long long totalTime;
		//-----------------------------------------------------------------INT VALUES-----------------------------------------------------------------------------------------

		//This will run if the user wants to use integers
		if (!useFloats) {
			std::vector<int> kernelOutput(inputElements);//Used to store the output of the kernel

			//Create a buffer to hold all of the tempurature values
			cl::Buffer buffer_tempuratures(context, CL_MEM_READ_ONLY, inputSize);
			queue.enqueueWriteBuffer(buffer_tempuratures, CL_TRUE, 0, inputSize, &weatherData.tempuratureInt[0]);
			auto bufferInitalisationFinish = std::chrono::high_resolution_clock::now(); //Ends the timer for intalising the buffers

			std::cout << "\nBuffers intialisation time (NS) = " << std::chrono::duration_cast<std::chrono::nanoseconds>(bufferInitalisationFinish - bufferInitalisationStart).count() << std::endl; //Outputs buffer initalisation time

			//Min------------------------------------------------------------------------------------------------------------------
			auto minStart = std::chrono::high_resolution_clock::now(); //Starts the timer for the min
			cl::Kernel kernel_min = cl::Kernel(program, "minValue"); //Creates a kernal variable for the minValue kernel
			//Sets the arguments in the kernel. arguement 2 (number 3) is used for local memory
			kernel_min.setArg(0, buffer_tempuratures);
			kernel_min.setArg(1, buffer_Output);
			kernel_min.setArg(2, cl::Local(workgroupSize * sizeof(int)));//local memory size
			queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize)); //Execututes the kernel
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]); //Reads the buffer into the vector

			int minTemp = kernelOutput[0]; //Store the min temp
			auto minFinish = std::chrono::high_resolution_clock::now(); //Stop the execution timer for the min temp

			//Outputs the min value and the execution time
			std::cout << "\nMin = " << minTemp << std::endl;
			minTime = std::chrono::duration_cast<std::chrono::nanoseconds>(minFinish - minStart).count();
			std::cout << "Execution time (NS) = " << minTime << std::endl;

			//Max------------------------------------------------------------------------------------------------------------------
			auto maxStart = std::chrono::high_resolution_clock::now(); //Starts the timer for the max
			cl::Kernel kernel_max = cl::Kernel(program, "maxValue");  //Creates a kernal variable for the maxValue kernel
			//Sets the arguments in the kernel. arguement 2 (number 3) is used for local memory
			kernel_max.setArg(0, buffer_tempuratures);
			kernel_max.setArg(1, buffer_Output);
			kernel_max.setArg(2, cl::Local(workgroupSize * sizeof(int)));//local memory size
			queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));//Execututes the kernel
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]);//Reads the buffer into the vector

			int maxTemp = kernelOutput[0]; //Store the max temp
			auto maxFinish = std::chrono::high_resolution_clock::now(); //Stop the execution timer for the max temp

			//Outputs the max value and the execution time
			std::cout << "\nMax = " << maxTemp << std::endl;
			maxTime = std::chrono::duration_cast<std::chrono::nanoseconds>(maxFinish - maxStart).count();
			std::cout << "Execution time (NS) = " << maxTime << std::endl;

			//Mean------------------------------------------------------------------------------------------------------------------
			//Read comments above
			auto meanStart = std::chrono::high_resolution_clock::now();
			cl::Kernel kernel_sum = cl::Kernel(program, "sum");
			kernel_sum.setArg(0, buffer_tempuratures);
			kernel_sum.setArg(1, buffer_Output);
			kernel_sum.setArg(2, cl::Local(workgroupSize * sizeof(int)));//local memory size
			queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]);

			float meanTemp = float(kernelOutput[0]) / originalSize;
			auto meanFinish = std::chrono::high_resolution_clock::now();

			std::cout << "\nMean = " << meanTemp << std::endl;
			meanTime = std::chrono::duration_cast<std::chrono::nanoseconds>(meanFinish - meanStart).count();
			std::cout << "Execution time (NS) = " << meanTime << std::endl;

			//SD------------------------------------------------------------------------------------------------------------------
			auto SDStart = std::chrono::high_resolution_clock::now();
			vector<int> meanValues(inputElements, meanTemp); //Creates a vector of the size of inputElements with all of the values set to the mean
			//Creates a mean buffer and then writes the vector we just created to it
			cl::Buffer buffer_mean(context, CL_MEM_READ_WRITE, inputSize);
			queue.enqueueWriteBuffer(buffer_mean, CL_TRUE, 0, inputSize, &meanValues[0]);

			//Creates a kernel for the variation from the mean squared and assigns it's arguments
			cl::Kernel kernel_VFMS = cl::Kernel(program, "variation");
			kernel_VFMS.setArg(0, buffer_tempuratures);
			kernel_VFMS.setArg(1, buffer_mean);
			kernel_VFMS.setArg(2, buffer_Output);
			kernel_VFMS.setArg(3, originalSize);
			queue.enqueueNDRangeKernel(kernel_VFMS, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize)); //Executes the buffer

			//Sums all of the values in the buffer outputted from VFMS as described in mean
			kernel_sum.setArg(0, buffer_Output);
			kernel_sum.setArg(1, buffer_Output2);
			kernel_sum.setArg(2, cl::Local(workgroupSize * sizeof(int)));//local memory size
			queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));
			queue.enqueueReadBuffer(buffer_Output2, CL_TRUE, 0, inputSize, &kernelOutput[0]);

			float SDTemp = (sqrt(kernelOutput[0] / originalSize)); //Final calculations for caluclating the standard Deviation
			auto SDFinish = std::chrono::high_resolution_clock::now();

			//Outputs the standard deviation and the execution time
			std::cout << "\nStandard Deviation = " << SDTemp << std::endl;
			SDTime = std::chrono::duration_cast<std::chrono::nanoseconds>(SDFinish - SDStart).count();
			std::cout << "Execution time (NS) = " << SDTime << std::endl;

			//Median global memory------------------------------------------------------------------------------------------------------------------
			//Similar to above
			auto medianGlobalStart = std::chrono::high_resolution_clock::now();
			cl::Kernel kernel_sort = cl::Kernel(program, "parallelSelection");
			kernel_sort.setArg(0, buffer_tempuratures);
			kernel_sort.setArg(1, buffer_Output);
			queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]);

			// because we added padding this will get rid of the number of 0's we added in the padding by finding the first one and then removing the next 300 values due to the vector being sorted
			for (int i = 0; i < inputSize; i++) {
				if (kernelOutput[i] == 0) {
					kernelOutput.erase(kernelOutput.begin() + i, kernelOutput.begin() + i + (inputElements - originalSize));
					break;
				}
			}

			//Self explanitory calculations
			int median = kernelOutput[kernelOutput.size() / 2];
			int firstQ = kernelOutput[kernelOutput.size() / 4];
			int thirdQ = kernelOutput[kernelOutput.size() / 4 * 3];
			int IQRange = (thirdQ - firstQ);
			auto medianGlobalFinish = std::chrono::high_resolution_clock::now();
			medianGlobalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(medianGlobalFinish - medianGlobalStart).count();

			//Median local memory------------------------------------------------------------------------------------------------------------------
			//Similar to above but done on local memory so therefore quicker
			auto medianLocalStart = std::chrono::high_resolution_clock::now();
			cl::Kernel kernel_sort_local = cl::Kernel(program, "parallelSelectionLocal");
			kernel_sort_local.setArg(0, buffer_tempuratures);
			kernel_sort_local.setArg(1, buffer_Output);
			kernel_sort_local.setArg(2, cl::Local(workgroupSize * sizeof(int)));//local memory size
			queue.enqueueNDRangeKernel(kernel_sort_local, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]);

			for (int i = 0; i < inputSize; i++) {
				if (kernelOutput[i] == 0) {
					kernelOutput.erase(kernelOutput.begin() + i, kernelOutput.begin() + i + (inputElements - originalSize));
					break;
				}
			}

			median = kernelOutput[kernelOutput.size() / 2];
			firstQ = kernelOutput[kernelOutput.size() / 4];
			thirdQ = kernelOutput[kernelOutput.size() / 4 * 3];
			IQRange = (thirdQ - firstQ);
			auto medianLocalFinish = std::chrono::high_resolution_clock::now();
			medianLocalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(medianLocalFinish - medianLocalStart).count();

			//Outputs the values calculated
			std::cout << "\nMedian = " << median << std::endl;
			std::cout << "Lower Quartile = " << firstQ << std::endl;
			std::cout << "Upper Quartile = " << thirdQ << std::endl;
			std::cout << "Interquartile Range = " << IQRange << std::endl;
			std::cout << "Execution time global (NS) = " << medianGlobalTime << std::endl;
			std::cout << "Execution time local (NS) = " << medianLocalTime << std::endl;
		}
		//-----------------------------------------------------------------FLOAT VALUES-----------------------------------------------------------------------------------------

		//If the user wishes to use the float values
		//This part is exactly the same as the code descirbed previously however it is modified to use floats and their corrisponding kernels
		else {

			std::vector<float> kernelOutput(inputElements);//Used to store the output of the kernel

			//Create a buffer to hold all of the tempurature values
			cl::Buffer buffer_tempuratures(context, CL_MEM_READ_ONLY, inputSize);
			queue.enqueueWriteBuffer(buffer_tempuratures, CL_TRUE, 0, inputSize, &weatherData.tempuratureFloat[0]);
			auto bufferInitalisationFinish = std::chrono::high_resolution_clock::now();

			std::cout << "\nBuffers intialisation time (NS) = " << std::chrono::duration_cast<std::chrono::nanoseconds>(bufferInitalisationFinish - bufferInitalisationStart).count() << std::endl;

			//Min------------------------------------------------------------------------------------------------------------------
			auto minStart = std::chrono::high_resolution_clock::now(); //Starts the timer for the min
			cl::Kernel kernel_min = cl::Kernel(program, "minValueFloat"); //Creates a kernal variable for the minValue kernel
			//Sets the arguments in the kernel. arguement 2 (number 3) is used for local memory
			kernel_min.setArg(0, buffer_tempuratures);
			kernel_min.setArg(1, buffer_Output);
			kernel_min.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size
			queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize)); //Execututes the kernel
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]); //Reads the buffer into the vector

			float minTemp = kernelOutput[0]; //Store the min temp
			auto minFinish = std::chrono::high_resolution_clock::now(); //Stop the execution timer for the min temp

			//Outputs the min value and the execution time
			std::cout << "\nMin = " << minTemp << std::endl;
			minTime = std::chrono::duration_cast<std::chrono::nanoseconds>(minFinish - minStart).count();
			std::cout << "Execution time (NS) = " << minTime << std::endl;

			//Max------------------------------------------------------------------------------------------------------------------
			auto maxStart = std::chrono::high_resolution_clock::now(); //Starts the timer for the max
			cl::Kernel kernel_max = cl::Kernel(program, "maxValueFloat");  //Creates a kernal variable for the maxValue kernel
			//Sets the arguments in the kernel. arguement 2 (number 3) is used for local memory
			kernel_max.setArg(0, buffer_tempuratures);
			kernel_max.setArg(1, buffer_Output);
			kernel_max.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size
			queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));//Execututes the kernel
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]);//Reads the buffer into the vector

			float maxTemp = kernelOutput[0]; //Store the max temp
			auto maxFinish = std::chrono::high_resolution_clock::now(); //Stop the execution timer for the max temp

			//Outputs the max value and the execution time
			std::cout << "\nMax = " << maxTemp << std::endl;
			maxTime = std::chrono::duration_cast<std::chrono::nanoseconds>(maxFinish - maxStart).count();
			std::cout << "Execution time (NS) = " << maxTime << std::endl;

			//Mean------------------------------------------------------------------------------------------------------------------
			//Read comments above
			auto meanStart = std::chrono::high_resolution_clock::now();
			cl::Kernel kernel_sum = cl::Kernel(program, "sumFloat");
			kernel_sum.setArg(0, buffer_tempuratures);
			kernel_sum.setArg(1, buffer_Output);
			kernel_sum.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size
			queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]);

			float meanTemp = float(kernelOutput[0]) / originalSize;
			auto meanFinish = std::chrono::high_resolution_clock::now();

			std::cout << "\nMean = " << meanTemp << std::endl;
			meanTime = std::chrono::duration_cast<std::chrono::nanoseconds>(meanFinish - meanStart).count();
			std::cout << "Execution time (NS) = " << meanTime << std::endl;

			//SD------------------------------------------------------------------------------------------------------------------
			auto SDStart = std::chrono::high_resolution_clock::now();
			vector<float> meanValues(inputElements, meanTemp); //Creates a vector of the size of inputElements with all of the values set to the mean
			//Creates a mean buffer and then writes the vector we just created to it
			cl::Buffer buffer_mean(context, CL_MEM_READ_WRITE, inputSize);
			queue.enqueueWriteBuffer(buffer_mean, CL_TRUE, 0, inputSize, &meanValues[0]);

			//Creates a kernel for the variation from the mean squared and assigns it's arguments
			cl::Kernel kernel_VFMS = cl::Kernel(program, "variationFloat");
			kernel_VFMS.setArg(0, buffer_tempuratures);
			kernel_VFMS.setArg(1, buffer_mean);
			kernel_VFMS.setArg(2, buffer_Output);
			kernel_VFMS.setArg(3, originalSize);
			queue.enqueueNDRangeKernel(kernel_VFMS, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize)); //Executes the buffer

			//Sums all of the values in the buffer outputted from VFMS as described in mean
			kernel_sum.setArg(0, buffer_Output);
			kernel_sum.setArg(1, buffer_Output2);
			kernel_sum.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size
			queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));
			queue.enqueueReadBuffer(buffer_Output2, CL_TRUE, 0, inputSize, &kernelOutput[0]);

			float SDTemp = (sqrt(kernelOutput[0] / originalSize)); //Final calculations for caluclating the standard Deviation
			auto SDFinish = std::chrono::high_resolution_clock::now();

			//Outputs the standard deviation and the execution time
			std::cout << "\nStandard Deviation = " << SDTemp << std::endl;
			SDTime = std::chrono::duration_cast<std::chrono::nanoseconds>(SDFinish - SDStart).count();
			std::cout << "Execution time (NS) = " << SDTime << std::endl;

			//Median global memory------------------------------------------------------------------------------------------------------------------
				//Similar to above
			auto medianGlobalStart = std::chrono::high_resolution_clock::now();
			cl::Kernel kernel_sort = cl::Kernel(program, "parallelSelectionFloat");
			kernel_sort.setArg(0, buffer_tempuratures);
			kernel_sort.setArg(1, buffer_Output);
			queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]);

			// because we added padding this will get rid of the number of 0's we added in the padding by finding the first one and then removing the next 300 values due to the vector being sorted
			for (int i = 0; i < inputSize; i++) {
				if (kernelOutput[i] == 0) {
					kernelOutput.erase(kernelOutput.begin() + i, kernelOutput.begin() + i + (inputElements - originalSize));
					break;
				}
			}

			//Self explanitory calculations
			float median = kernelOutput[kernelOutput.size() / 2];
			float firstQ = kernelOutput[kernelOutput.size() / 4];
			float thirdQ = kernelOutput[kernelOutput.size() / 4 * 3];
			float IQRange = (thirdQ - firstQ);
			auto medianGlobalFinish = std::chrono::high_resolution_clock::now();
			medianGlobalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(medianGlobalFinish - medianGlobalStart).count();

			//Median local memory------------------------------------------------------------------------------------------------------------------
				//Similar to above but done on local memory so therefore quicker
			auto medianLocalStart = std::chrono::high_resolution_clock::now();
			cl::Kernel kernel_sort_local = cl::Kernel(program, "parallelSelectionLocalFloat");
			kernel_sort_local.setArg(0, buffer_tempuratures);
			kernel_sort_local.setArg(1, buffer_Output);
			kernel_sort_local.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size
			queue.enqueueNDRangeKernel(kernel_sort_local, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(workgroupSize));
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, inputSize, &kernelOutput[0]);

			for (int i = 0; i < inputSize; i++) {
				if (kernelOutput[i] == 0) {
					kernelOutput.erase(kernelOutput.begin() + i, kernelOutput.begin() + i + (inputElements - originalSize));
					break;
				}
			}

			auto medianLocalFinish = std::chrono::high_resolution_clock::now();
			medianLocalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(medianLocalFinish - medianLocalStart).count();

			//Outputs the values calculated
			std::cout << "\nMedian = " << median << std::endl;
			std::cout << "Lower Quartile = " << firstQ << std::endl;
			std::cout << "Upper Quartile = " << thirdQ << std::endl;
			std::cout << "Interquartile Range = " << IQRange << std::endl;
			std::cout << "Execution time global (NS) = " << medianGlobalTime << std::endl;
			std::cout << "Execution time local (NS) = " << medianLocalTime << std::endl;
		}

		//Finishes the toal executition time of the algorithm, stores it and displays it
		auto Finish = std::chrono::high_resolution_clock::now();
		totalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(Finish - Start).count();
		cout << "\nTotal execution time (NS) = " << totalTime << std::endl;

		std::getchar();
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}