package gqserver.api;

import java.io.IOException;
import java.io.Serializable;

public interface Packet extends Serializable {

    default void onServerReceive(ServerClient serverClient) throws IOException {}

}
