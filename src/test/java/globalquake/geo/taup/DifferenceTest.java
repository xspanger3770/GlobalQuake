package globalquake.geo.taup;

public class DifferenceTest {

    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();

        double ang = TauPTravelTimeCalculator.toAngle(8372.6);

        for (double depth = 0; depth < 50; depth += 0.5) {
            System.out.printf("%.1fkm: %.3fs%n",depth, TauPTravelTimeCalculator.getPWaveTravelTime(depth, ang) - TravelTimeTableOld.getPWaveTravelTime(depth, ang));
        }
    }

}
