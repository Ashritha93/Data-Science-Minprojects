/* Welcome to the SQL mini project. For this project, you will use
Springboard' online SQL platform, which you can log into through the
following link:

https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

Note that, if you need to, you can also download these tables locally.

In the mini project, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */



/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */
SELECT *
FROM Facilities
WHERE Facilities.membercost > 0

/* Tennis court 1 & 2, Massage Rooms 1 & 2, Squash court */

/* Q2: How many facilities do not charge a fee to members? */
SELECT *
FROM Facilities
WHERE Facilities.membercost =0

/* Results to only four rows */

/* Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */
SELECT facid, name, membercost, monthlymaintenance FROM Facilities WHERE Facilities.membercost <= (Facilities.monthlymaintenance*0.2)

/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */
SELECT *
FROM Facilities
WHERE Facilities.facid
IN ( 1, 5 )

/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */

SELECT facid, name, monthlymaintenance,
	CASE
		WHEN monthlymaintenance >= 100 THEN 'expensive'
		ELSE 'cheap'
	END AS label
	FROM Facilities

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */

/* Method 1 */
SELECT firstname, surname, joindate FROM Members ORDER BY Members.joindate DESC LIMIT 3 
/* Method 2 */
SELECT firstname, surname, joindate FROM Members mem1 
WHERE 3 > ( select count(1) from Members mem2 where mem2.memid > mem1.memid )

/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

SELECT DISTINCT CONCAT( mem.firstname, ' ', mem.surname ) AS fullname
FROM Members mem
INNER JOIN Bookings boo ON mem.memid = boo.memid
INNER JOIN Facilities fac ON boo.facid = fac.facid
WHERE fac.facid
IN ( 0, 1 ) 

/* Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

CREATE VIEW view1 AS
SELECT fac.name AS facilityName, CONCAT( mem.firstname, ' ', mem.surname ) AS fullName,
CASE
WHEN boo.memid =0 THEN boo.slots * fac.guestcost
ELSE boo.slots * fac.membercost
END AS cost
FROM Members mem
INNER JOIN Bookings boo ON mem.memid = boo.memid
INNER JOIN Facilities fac ON boo.facid = fac.facid
WHERE boo.starttime LIKE '2012-09-14%'
ORDER BY cost DESC;

SELECT * FROM view1 where view1.cost > 30
/* Not able to create views in sql.springboard.com */

/* Q9: This time, produce the same result as in Q8, but using a subquery. */
SELECT * FROM (
SELECT fac.name AS facilityName, CONCAT( mem.firstname, ' ', mem.surname ) AS fullName,
CASE
WHEN boo.memid =0 THEN boo.slots * fac.guestcost
ELSE boo.slots * fac.membercost
END AS cost
FROM Members mem
INNER JOIN Bookings boo ON mem.memid = boo.memid
INNER JOIN Facilities fac ON boo.facid = fac.facid
WHERE boo.starttime LIKE '2012-09-14%'
ORDER BY cost DESC) sub
WHERE sub.cost > 30

/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

SELECT b.facid, b.name, SUM(b.guestRev) + SUM(b.memRev) as cost
FROM (SELECT a.facid as facid, a.name as name, a.guestSlots * a.guestcost as guestRev, a.memSlots * a.membercost as memRev
FROM (SELECT fac.facid as facid, fac.name as name,
CASE
WHEN boo.memid = 0 THEN boo.slots
ELSE 0
END AS guestSlots, fac.guestcost as guestcost,
CASE 
WHEN boo.memid != 0 THEN boo.slots
ELSE 0
END AS memSlots, fac.membercost as membercost
FROM Facilities fac
INNER JOIN Bookings boo ON fac.facid = boo.facid 
INNER JOIN Members mem ON mem.memid = boo.memid) a) b
GROUP BY b.facid
ORDER BY cost DESC
