#include <iostream>
#include <cstring>      // For memset
#include <sys/socket.h> // For socket, connect
#include <netinet/in.h> // For sockaddr_in
#include <unistd.h>     // For close
#include <arpa/inet.h>
#include <cmath>

#include "travel_table.hpp"

std::unique_ptr<TravelTable> loaded_travel_table;

int createSocket(){
    // Create a socket
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return -1;
    }

    // Define the server address and port
    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(12345); // Use the port of the server you want to connect to
    serverAddress.sin_addr.s_addr = inet_addr("0.0.0.0"); // Use the IP address of the server you want to connect to

    // Connect to the server
    if (connect(clientSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) == -1) {
        std::cerr << "Error connecting to server: " << strerror(errno) << std::endl;
        close(clientSocket);
        return -1;
    }

    std::cout << "Connected to the server" << std::endl;

    return clientSocket;
}

void fill(const std::unique_ptr<TravelTable>& travel_table, table& table, int fd, std::string wave, double minAng, double maxAng){
    char buff[64];
    char input[64];

    for (int angI = 0; angI <= static_cast<int>(ceil((maxAng - minAng) / travel_table->angularResolution)); angI ++) {
        double ang = minAng + angI * travel_table->angularResolution;
        for (int depthI = 0; depthI <= static_cast<int>(ceil(travel_table->maxDepth / travel_table->depthResolution)); depthI++) {
            double depth = depthI * travel_table->depthResolution;
            snprintf(buff, 64, "%s %f %f\n", wave.data(), ang, depth);
            write(fd, buff, strlen(buff));
            read(fd, input, 64);

            float time = atof(input);

            table[angI][depthI] = time;
        }
        std::cout << angI << std::endl;
    }
}

int createTravelTable(float maxDepth, float depthResolution, float angularResolution) {
    int fd = createSocket();
    if(fd == -1){
        return -1;
    }

    loaded_travel_table = std::make_unique<TravelTable>(maxDepth, depthResolution, angularResolution);
    fill(loaded_travel_table, loaded_travel_table->p_wave, fd, "P", 0, P_S_MAX_ANGLE);
    fill(loaded_travel_table, loaded_travel_table->s_wave, fd, "s", 0, P_S_MAX_ANGLE);
    fill(loaded_travel_table, loaded_travel_table->pkp_wave, fd, "pkp", PKP_MIN_ANGLE, PKP_MAX_ANGLE);

    close(fd);

    return 0;
}