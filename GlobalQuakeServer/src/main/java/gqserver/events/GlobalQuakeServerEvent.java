package gqserver.events;

public interface GlobalQuakeServerEvent {
    void run(GlobalQuakeServerEventListener eventListener);
}
