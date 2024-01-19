package globalquake.ui.globalquake.feature;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.earthquake.quality.QualityClass;
import globalquake.utils.GeoUtils;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.core.Settings;
import globalquake.utils.Scale;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

import java.awt.*;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Collection;
import java.util.List;
import java.util.Locale;

public class FeatureEarthquake extends RenderFeature<Earthquake> {

    private static final int ELEMENT_COUNT = 5 + 4;
    private final Collection<Earthquake> earthquakes;

    public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));

    public FeatureEarthquake(Collection<Earthquake> earthquakes) {
        super(ELEMENT_COUNT);
        this.earthquakes = earthquakes;
    }

    @Override
    public Collection<Earthquake> getElements() {
        return earthquakes;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<Earthquake> entity, RenderProperties renderProperties) {
        RenderElement elementPWave = entity.getRenderElement(0);
        RenderElement elementSWave = entity.getRenderElement(1);
        RenderElement elementPKPWave = entity.getRenderElement(2);
        RenderElement elementPKIKPWave = entity.getRenderElement(3);
        RenderElement elementCross = entity.getRenderElement(4);

        Earthquake e = entity.getOriginal();

        long age = GlobalQuake.instance.currentTimeMillis() - e.getOrigin();
        double pDist = TauPTravelTimeCalculator.getPWaveTravelAngle(e.getDepth(), age / 1000.0) / 360.0
                * GeoUtils.EARTH_CIRCUMFERENCE;
        double sDist = TauPTravelTimeCalculator.getSWaveTravelAngle(e.getDepth(), age / 1000.0) / 360.0
                * GeoUtils.EARTH_CIRCUMFERENCE;
        double pkpDist = TauPTravelTimeCalculator.getPKPWaveTravelAngle(e.getDepth(), age / 1000.0) / 360.0
                * GeoUtils.EARTH_CIRCUMFERENCE;
        double pkikpDist = TauPTravelTimeCalculator.getPKIKPWaveTravelAngle(e.getDepth(), age / 1000.0) / 360.0
                * GeoUtils.EARTH_CIRCUMFERENCE;

        renderer.createCircle(elementPWave.getPolygon(),
                entity.getOriginal().getLat(),
                entity.getOriginal().getLon(),
                Math.max(0, pDist), 0, GlobeRenderer.QUALITY_HIGH);

        renderer.createCircle(elementSWave.getPolygon(),
                entity.getOriginal().getLat(),
                entity.getOriginal().getLon(),
                Math.max(0, sDist), 0, GlobeRenderer.QUALITY_HIGH);

        renderer.createCircle(elementPKPWave.getPolygon(),
                entity.getOriginal().getLat(),
                entity.getOriginal().getLon(),
                Math.max(0, pkpDist), 0, GlobeRenderer.QUALITY_HIGH);

        renderer.createCircle(elementPKIKPWave.getPolygon(),
                entity.getOriginal().getLat(),
                entity.getOriginal().getLon(),
                Math.max(0, pkikpDist), 0, GlobeRenderer.QUALITY_HIGH);

        renderer.createCross(elementCross.getPolygon(),
                entity.getOriginal().getLat(),
                entity.getOriginal().getLon(), renderer
                        .pxToDeg(16, renderProperties), 45.0);


        if(e.getCluster() != null && e.getCluster().getPreviousHypocenter() != null && e.getCluster().getPreviousHypocenter().polygonConfidenceIntervals != null) {
            List<PolygonConfidenceInterval> polygonConfidenceIntervals = e.getCluster().getPreviousHypocenter().polygonConfidenceIntervals;
            for (int i = 0; i < polygonConfidenceIntervals.size(); i++) {
                PolygonConfidenceInterval polygonConfidenceInterval = polygonConfidenceIntervals.get(i);
                createConfidencePolygon(entity.getRenderElement(5 + i), polygonConfidenceInterval, entity.getOriginal().getLat(), entity.getOriginal().getLon());
            }
        }
    }

    private void createConfidencePolygon(RenderElement renderElement, PolygonConfidenceInterval polygonConfidenceInterval, double lat, double lon) {
        renderElement.getPolygon().reset();

        double step = 360.0 / polygonConfidenceInterval.n();

        for (int i = 0; i < polygonConfidenceInterval.n() + 1; i++) {
            double ang = polygonConfidenceInterval.offset() + step * i;
            double[] latLon = GeoUtils.moveOnGlobe(lat, lon, polygonConfidenceInterval.lengths().get(i % polygonConfidenceInterval.n()), ang);
            Vector3D vector3D = new Vector3D(
                    GlobeRenderer.getX_3D(latLon[0], latLon[1], 0),
                    GlobeRenderer.getY_3D(latLon[0], latLon[1], 0),
                    GlobeRenderer.getZ_3D(latLon[0], latLon[1], 0));

            renderElement.getPolygon().addPoint(vector3D);
        }

        renderElement.getPolygon().finish();
    }

    @Override
    public boolean needsUpdateEntities() {
        return true;
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<Earthquake> entity, boolean propertiesChanged) {
        return true;
    }

    @Override
    public boolean needsProject(RenderEntity<Earthquake> entity, boolean propertiesChanged) {
        return true;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<Earthquake> entity, RenderProperties renderProperties) {
        for (int i = 0; i < ELEMENT_COUNT; i++) {
            RenderElement elementPWave = entity.getRenderElement(i);
            elementPWave.getShape().reset();
            elementPWave.shouldDraw = renderer.project3D(elementPWave.getShape(), elementPWave.getPolygon(), true, renderProperties);
        }
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<Earthquake> entity, RenderProperties renderProperties) {
        float thicknessMultiplier = (float) Math.max(0.3, Math.min(1.6, entity.getOriginal().getMag() / 5.0));
        RenderElement elementPWave = entity.getRenderElement(0);
        RenderElement elementSWave = entity.getRenderElement(1);
        RenderElement elementPKPWave = entity.getRenderElement(2);
        RenderElement elementPKIKPWave = entity.getRenderElement(3);

        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        if(Settings.confidencePolygons) {
            for (int i = 5; i < 9; i++) {
                RenderElement elementConfidencePolygon = entity.getRenderElement(i);
                if (elementConfidencePolygon.shouldDraw) {
                    graphics.setStroke(new BasicStroke(3.0f));
                    graphics.setColor(polygonColor(i - 5));
                    graphics.draw(elementConfidencePolygon.getShape());
                }
            }
        }

        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        if (elementPWave.shouldDraw) {
            graphics.setColor(alphaColor(Color.BLUE, getAlphaMul(entity.getOriginal())));
            graphics.setStroke(new BasicStroke(4.0f * thicknessMultiplier));
            graphics.draw(elementPWave.getShape());
        }

        if (elementSWave.shouldDraw) {
            graphics.setColor(alphaColor(getColorSWave(entity.getOriginal().getMag()), getAlphaMul(entity.getOriginal())));
            graphics.setStroke(new BasicStroke(4.0f * thicknessMultiplier));
            graphics.draw(elementSWave.getShape());
        }

        if (Settings.displayCoreWaves) {
            if (elementPKPWave.shouldDraw) {
                graphics.setColor(Color.MAGENTA);
                graphics.setStroke(new BasicStroke(4.0f * thicknessMultiplier));
                graphics.draw(elementPKPWave.getShape());
            }

            if (elementPKIKPWave.shouldDraw) {
                graphics.setColor(Color.GREEN);
                graphics.setStroke(new BasicStroke(1.0f));
                graphics.draw(elementPKIKPWave.getShape());
            }
        }

        RenderElement elementCross = entity.getRenderElement(4);
        if (elementCross.shouldDraw) {
            boolean isUncertain = isUncertain(entity.getOriginal().getHypocenter());

            if((System.currentTimeMillis() / 500) % 2 == 0 && !isUncertain) {
                graphics.setStroke(new BasicStroke(4f));
                graphics.setColor(getCrossColor(entity.getOriginal().getMag()));
                graphics.draw(elementCross.getShape());
            }

            var point3D = GlobeRenderer.createVec3D(getCenterCoords(entity));
            var centerPonint = renderer.projectPoint(point3D, renderProperties);

            if(isUncertain && (System.currentTimeMillis() / 500) % 2 == 0){
                graphics.setColor(Color.WHITE);
                graphics.setFont(new Font("Calibri", Font.BOLD, 32));
                String str = "?";
                graphics.drawString(str, (int) (centerPonint.x - graphics.getFontMetrics().stringWidth(str) / 2), (int) (centerPonint.y + 10));
            }

            String str = "M%s".formatted(f1d.format(entity.getOriginal().getMag()));

            graphics.setColor(Color.WHITE);
            graphics.setFont(new Font("Calibri", Font.BOLD, 16));
            graphics.drawString(str, (int) (centerPonint.x - graphics.getFontMetrics().stringWidth(str) / 2), (int) (centerPonint.y - 18));

            Cluster cluster = entity.getOriginal().getCluster();
            if (cluster != null) {
                Hypocenter hypocenter = cluster.getPreviousHypocenter();

                if (hypocenter != null) {
                    str = "%s".formatted(
                            Settings.getSelectedDistanceUnit().format(hypocenter.depth, 1)
                    );

                    graphics.drawString(str, (int) (centerPonint.x - graphics.getFontMetrics().stringWidth(str) / 2), (int) (centerPonint.y + 29));
                }
            }
        }

        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
    }

    private boolean isUncertain(Hypocenter hypocenter) {
        return hypocenter.quality != null && hypocenter.quality.getSummary() == QualityClass.D;
    }

    private Color alphaColor(Color color, double mul) {
        return new Color(color.getRed(), color.getGreen(), color.getBlue(), (int) (255.0 * mul));
    }

    private double getAlphaMul(Earthquake original) {
        double ageMins = (GlobalQuake.instance.currentTimeMillis() - original.getOrigin()) / (1000.0 * 60.0);
        double limit = waveDisplayTimeMinutes(original.getMag(), original.getDepth());

        return Math.max(0, Math.min(1.0, 2.0 - 2.0 * ageMins / limit));
    }

    private double waveDisplayTimeMinutes(double mag, double depth) {
        return 2.0 + 0.01 * Math.pow(mag + EarthquakeAnalysis.getDepthCorrection(depth), 4);
    }

    private Color polygonColor(int i) {
        if(i == 0){
            return Color.blue;
        }
        if(i == 1){
            return Color.green;
        }
        if(i == 2){
            return Color.yellow;
        }
        return Color.red;
    }

    private Color getColorSWave(double mag) {
        double weight = Math.max(0, Math.min(1, (mag - 2.0) / 4.0));
        return Scale.interpolateColors(Color.yellow, Color.red, weight);
    }

    public static Color getCrossColor(double mag) {
        if (mag < 3) {
            return Color.white;
        }
        if (mag < 4) {
            return Color.green;
        }
        if (mag < 5) {
            return Color.yellow;
        }
        if (mag < 6) {
            return Color.orange;
        }
        if (mag < 7) {
            return Color.red;
        }
        return Color.magenta;
    }


    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return new Point2D(((Earthquake) (entity.getOriginal())).getLat(), ((Earthquake) (entity.getOriginal())).getLon());
    }
}
