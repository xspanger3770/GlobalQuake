package globalquake.ui.stationselect;

import globalquake.database.Network;
import globalquake.database.Station;
import globalquake.database.StationDatabase;
import globalquake.ui.globe.GlobePanel;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;

public class StationSelectPanel extends GlobePanel {

    private final StationDatabase stationDatabase;
    private final MonitorableCopyOnWriteArrayList<Station> allStationsList = new MonitorableCopyOnWriteArrayList<>();
    private final StationSelectFrame stationSelectFrame;
    public boolean showUnavailable;

    private Point dragStart;
    private Point dragEnd;
    private Rectangle dragRectangle;

    public Rectangle getDragRectangle() {
        return dragRectangle;
    }

    public StationSelectPanel(StationSelectFrame stationSelectFrame, StationDatabase stationDatabase) {
        this.stationDatabase = stationDatabase;
        this.stationSelectFrame = stationSelectFrame;
        updateAllStations();
        getRenderer().addFeature(new FeatureSelectableStation(allStationsList, this));

        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (stationSelectFrame.getDragMode().equals(DragMode.NONE)) {
                    return;
                }

                dragEnd = e.getPoint();

                if(dragStart != null) {
                    Rectangle rectangle = new Rectangle(dragStart);
                    rectangle.add(dragEnd);
                    dragRectangle = rectangle;
                }
            }
        });

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (stationSelectFrame.getDragMode().equals(DragMode.NONE)) {
                    return;
                }

                if(e.getButton() == MouseEvent.BUTTON3){
                    stationSelectFrame.setDragMode(DragMode.NONE);
                    dragEnd = null;
                    dragRectangle = null;
                    return;
                } else if(e.getButton() != MouseEvent.BUTTON1){
                    return;
                }

                dragEnd = null;
                dragRectangle = null;

                dragStart = e.getPoint();
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                if (stationSelectFrame.getDragMode().equals(DragMode.NONE)) {
                    return;
                }

                if(dragEnd == null || e.getButton() != MouseEvent.BUTTON1){
                    return;
                }

                dragEnd = e.getPoint();
                if(dragStart != null) {
                    Rectangle rectangle = new Rectangle(dragStart);
                    rectangle.add(dragEnd);
                    dragRectangle = rectangle;
                }

                fireDragEvent();

                dragEnd = null;
                dragStart = null;
                dragRectangle = null;
            }
        });
    }

    private void fireDragEvent() {

    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        Graphics2D g = (Graphics2D) gr;
        if(stationSelectFrame.getDragMode() == DragMode.NONE){
            return;
        }

        if(dragRectangle != null){
            g.setColor(stationSelectFrame.getDragMode() == DragMode.SELECT ? Color.green:Color.red);
            g.setStroke(new BasicStroke(2f));
            g.draw(dragRectangle);
        }
    }

    @Override
    public boolean interactionAllowed() {
        return stationSelectFrame.getDragMode().equals(DragMode.NONE);
    }

    public void updateAllStations() {
        List<Station> stations = new ArrayList<>();
        stationDatabase.getDatabaseReadLock().lock();
        try{
            for(Network network:stationDatabase.getNetworks()){
                stations.addAll(network.getStations().stream().filter(station -> showUnavailable || station.hasAvailableChannel()).toList());
            }

            allStationsList.clear();
            allStationsList.addAll(stations);
            System.out.println(allStationsList.size());
        }finally {
            stationDatabase.getDatabaseReadLock().unlock();
        }
    }
}
