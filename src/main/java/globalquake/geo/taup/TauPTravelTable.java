package globalquake.geo.taup;

import edu.sc.seis.TauP.Arrival;
import edu.sc.seis.TauP.TauModelException;
import edu.sc.seis.TauP.TauP_Time;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class TauPTravelTable implements Serializable {


    public static final double P_S_MIN_ANGLE = 0;
    public static final double P_S_MAX_ANGLE = 150;

    public static final double PKIKP_MIN_ANGLE = 0;
    public static final double PKIKP_MAX_ANGLE = 180;

    public static final double PKP_MIN_ANGLE = 140;
    public static final double PKP_MAX_ANGLE = 180;
    public static final String MODEL_NAME = "iasp91";

    public float[][] p_travel_table;
    public float[][] s_travel_table;

    public float[][] pkikp_travel_table;
    public float[][] pkp_travel_table;

    public TauPTravelTable() throws TauModelException, IOException {
        TauP_Time timeToolGlobal = new TauP_Time();
        timeToolGlobal.loadTauModel(MODEL_NAME);

        p_travel_table = createArray(P_S_MIN_ANGLE, P_S_MAX_ANGLE);
        fill(p_travel_table, timeToolGlobal, "p,P,pP,Pn,PcP,Pdiff", P_S_MIN_ANGLE, P_S_MAX_ANGLE);

        s_travel_table = createArray(P_S_MIN_ANGLE, P_S_MAX_ANGLE);
        fill(s_travel_table, timeToolGlobal, "s,S,sS,Sn,ScS,Sdiff", P_S_MIN_ANGLE, P_S_MAX_ANGLE);

        pkikp_travel_table = createArray(PKIKP_MIN_ANGLE, PKIKP_MAX_ANGLE);
        fill(pkikp_travel_table, timeToolGlobal, "PKIKP,PKiKP", PKIKP_MIN_ANGLE, PKIKP_MAX_ANGLE);

        pkp_travel_table = createArray(PKP_MIN_ANGLE, PKP_MAX_ANGLE);
        fill(pkp_travel_table, timeToolGlobal, "PKP", PKP_MIN_ANGLE, PKP_MAX_ANGLE);
    }

    @SuppressWarnings("CallToPrintStackTrace")
    private static void fill(float[][] array, TauP_Time timeModel, String phases, double minAngle, double maxAngle) {
        List<Double> depths = new ArrayList<>();

        for (double depth = 0; depth <= TauPTravelTimeCalculator.MAX_DEPTH; depth += TauPTravelTimeCalculator.DEPTH_RESOLUTION) {
            depths.add(depth);
        }

        depths.parallelStream().forEach(depth -> {
            try {
                TauP_Time timeTool = new TauP_Time(timeModel.getTauModelName());
                timeTool.parsePhaseList(phases);

                for (double ang = minAngle; ang <= maxAngle; ang += TauPTravelTimeCalculator.ANG_RESOLUTION) {
                    timeTool.setSourceDepth(depth);
                    timeTool.calculate(ang);
                    if (timeTool.getNumArrivals() > 0) {
                        Arrival arrival = timeTool.getArrival(0);
                        array[(int) (depth / TauPTravelTimeCalculator.DEPTH_RESOLUTION)][(int) ((ang - minAngle) / TauPTravelTimeCalculator.ANG_RESOLUTION)] = (float) arrival.getTime();
                    } else {
                        array[(int) (depth / TauPTravelTimeCalculator.DEPTH_RESOLUTION)][(int) ((ang - minAngle) / TauPTravelTimeCalculator.ANG_RESOLUTION)] = TauPTravelTimeCalculator.NO_ARRIVAL;
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
        });
    }

    private static float[][] createArray(double minAng, double maxAng) {
        int count = ((int) (TauPTravelTimeCalculator.MAX_DEPTH / TauPTravelTimeCalculator.DEPTH_RESOLUTION) + 1) * ((int) ((maxAng - minAng) / TauPTravelTimeCalculator.ANG_RESOLUTION) + 1);
        double size = (count * Float.BYTES) / (1024.0 * 1024.0);
        System.out.printf("Filling array size %,d (%.3fMB)%n", count, size);
        return new float[(int) (TauPTravelTimeCalculator.MAX_DEPTH / TauPTravelTimeCalculator.DEPTH_RESOLUTION) + 1][(int) ((maxAng - minAng) / TauPTravelTimeCalculator.ANG_RESOLUTION) + 1];
    }
}
