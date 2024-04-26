package globalquake.ui.stationselect;

import globalquake.core.database.Network;
import globalquake.core.database.Station;
import globalquake.core.database.StationDatabaseManager;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.core.Settings;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;

public class StationSelectPanel extends GlobePanel {

    private final StationDatabaseManager stationDatabaseManager;
    private final MonitorableCopyOnWriteArrayList<Station> allStationsList = new MonitorableCopyOnWriteArrayList<>();
    private final StationSelectFrame stationSelectFrame;
    private final FeatureSelectableStation featureSelectableStation;
    public boolean showUnavailable;

    private Point dragStart;
    private Point dragEnd;
    private Rectangle dragRectangle;

    public Rectangle getDragRectangle() {
        return dragRectangle;
    }

    public StationSelectPanel(StationSelectFrame stationSelectFrame, StationDatabaseManager stationDatabaseManager) {
        super(Settings.homeLat, Settings.homeLon);
        this.stationDatabaseManager = stationDatabaseManager;
        this.stationSelectFrame = stationSelectFrame;
        updateAllStations();
        getRenderer().addFeature(featureSelectableStation = new FeatureSelectableStation(allStationsList, this));

        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (stationSelectFrame.getDragMode().equals(DragMode.NONE)) {
                    return;
                }

                dragEnd = e.getPoint();

                if (dragStart != null) {
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

                if (e.getButton() == MouseEvent.BUTTON3) {
                    stationSelectFrame.setDragMode(DragMode.NONE);
                    dragEnd = null;
                    dragRectangle = null;
                    return;
                } else if (e.getButton() != MouseEvent.BUTTON1) {
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

                if (dragEnd == null || e.getButton() != MouseEvent.BUTTON1) {
                    return;
                }

                dragEnd = e.getPoint();
                if (dragStart != null) {
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

    @Override
    public void featuresClicked(ArrayList<RenderEntity<?>> clicked) {
        List<Station> clickedStations = new ArrayList<>();
        for (RenderEntity<?> renderEntity : clicked) {
            if (renderEntity.getOriginal() instanceof Station) {
                clickedStations.add((Station) renderEntity.getOriginal());
            }
        }

        if (clickedStations.isEmpty()) {
            return;
        }

        Station selectedStation;

        if (clickedStations.size() == 1) {
            selectedStation = clickedStations.get(0);
        } else {
            selectedStation = (Station) JOptionPane.showInputDialog(this, "Select station:", "Station selection",
                    JOptionPane.PLAIN_MESSAGE, null, clickedStations.toArray(), clickedStations.get(0));
        }

        if (selectedStation != null)
            new StationEditDialog(stationSelectFrame, selectedStation);
    }

    private void fireDragEvent() {
        stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try {
            List<Station> selected = getRenderer().getAllInside(featureSelectableStation, dragRectangle, getRenderer().getRenderProperties());
            if (stationSelectFrame.getDragMode().equals(DragMode.SELECT)) {
                selected.forEach(Station::selectBestChannel);
            } else {
                selected.forEach(station -> station.setSelectedChannel(null));
            }

            stationDatabaseManager.fireUpdateEvent();
        } finally {
            stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }
    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        Graphics2D g = (Graphics2D) gr;
        if (stationSelectFrame.getDragMode() == DragMode.NONE) {
            return;
        }

        g.setColor(Color.orange);
        g.setFont(new Font("Calibri", Font.BOLD, 18));
        String str = stationSelectFrame.getDragMode() == DragMode.SELECT ? "Drag to select region"
                : "Drag to deselect region";
        g.drawString(str, getWidth() / 2 - g.getFontMetrics().stringWidth(str) / 2, getHeight() - 8);

        if (dragRectangle != null) {
            g.setColor(stationSelectFrame.getDragMode() == DragMode.SELECT ? Color.green : Color.red);
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
        stationDatabaseManager.getStationDatabase().getDatabaseReadLock().lock();
        try {
            for (Network network : stationDatabaseManager.getStationDatabase().getNetworks()) {
                stations.addAll(network.getStations().stream().filter(station -> showUnavailable || station.hasAvailableChannel()).toList());
            }

            allStationsList.clear();
            allStationsList.addAll(stations);
        } finally {
            stationDatabaseManager.getStationDatabase().getDatabaseReadLock().unlock();
        }
    }
}
