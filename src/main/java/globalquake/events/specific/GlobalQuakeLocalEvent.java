package globalquake.events.specific;


import globalquake.events.GlobalQuakeLocalEventListener;

public interface GlobalQuakeLocalEvent {

    void run(GlobalQuakeLocalEventListener eventListener);
}
