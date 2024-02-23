package globalquake.core.earthquake.data;

import gqserver.api.packets.station.InputType;

public record MagnitudeReading(double magnitude, double distance, long eventAge, InputType inputType) {
}
