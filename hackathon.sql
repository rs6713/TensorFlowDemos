SELECT table_schema,table_name
FROM information_schema.tables
ORDER BY table_schema,table_name;

set search_path = musicbrainz;
/* a.ended whether artist still active */
/*
select a.name as "artist",  g.name as "gender", a.begin_date_year as "career start", a.end_date_year as "career end", a.end_date_year-a.begin_date_year as "career duration",
at.name as "artist type",
ar.name as "area", ar_t.name as "area type"
from artist as a 
left join gender as g on g.id=a.gender
left join artist_type as at on at.id=a.type
left join area as ar on a.area= ar.id
left join area_type as ar_t on ar_t.id = ar.type
where a.begin_date_year is not null and a.end_date_year is not null
;
*/
select * from
(select a.id as "id", count(ac.id)as "number releases", round(avg(ac.artist_count),2) as "avg size artists collabs",/*count(ac.id) as "number releases",*/
round(avg(rc.length),2) as "average song length"
from artist as a
left join artist_credit_name as acn on acn.artist=a.id
left join artist_credit as ac on ac.id=acn.artist_credit
left join recording as rc on rc.artist_credit = ac.id
where a.name != 'Various Artists' 
group by a.id) as ta join
(select a.id as "id", a.name as "artist",  g.name as "gender", a.begin_date_year as "career start", a.end_date_year as "career end", a.end_date_year-a.begin_date_year as "career duration",
at.name as "artist type",
ar.name as "area", ar_t.name as "area type"
from artist as a 
left join gender as g on g.id=a.gender
left join artist_type as at on at.id=a.type
left join area as ar on a.area= ar.id
left join area_type as ar_t on ar_t.id = ar.type
where a.begin_date_year is not null and a.end_date_year is not null) as tb
on ta.id = tb.id;


/*
select a.id as "artist id", count(ac.id)as "number releases", round(avg(ac.artist_count),2) as "avg size artists collabs",/*count(ac.id) as "number releases",*/
round(avg(rc.length),2) as "average song length"
from artist as a
left join artist_credit_name as acn on acn.artist=a.id
left join artist_credit as ac on ac.id=acn.artist_credit
left join recording as rc on rc.artist_credit = ac.id
where a.name != 'Various Artists' 
group by a.id
limit 100;
*/
/*
select a.id as "artist id", count(ac.id)as "number releases", round(avg(ac.artist_count),2) as "avg size artists collabs",/*count(ac.id) as "number releases",*/
 count(distinct(l.name)) as "number diff langs of releases",
 array_agg(re_c.country),array_agg(l.name),
 count(distinct(re_c.country)) as "number diff countries of releases",
round(avg(rc.length),2) as "average song length"
from artist as a
left join artist_credit_name as acn on acn.artist=a.id
left join artist_credit as ac on ac.id=acn.artist_credit
left join release as r on r.artist_credit = ac.id
left join language as l on r.language = l.id
left join recording as rc on rc.artist_credit = ac.id
left join release_country as re_c on re_c.release = r.id
where a.name != 'Various Artists' 
group by a.id
limit 100;
*/
/*
select r.name, r.length as "recording length", m.name as "medium",
mf.name as "medium_format", m.track_count as "track count",
ac.artist_count as "artist count", ac.name as "artist credit name",
rg.name as "release group name", rgt.name as "release type",
re.name as "release name", re_s.name as "release status",
a.name as "area", re_c.date_year as "release country year"
from recording as r
left join track as t on r.id = t.recording
left join medium as m on m.id = t.medium
left join medium_format as mf on m.format=mf.id
left join artist_credit as ac on r.artist_credit = ac.id
left join release_group as rg on rg.artist_credit = ac.id
left join release_group_primary_type as rgt on rg.type=rgt.id
left join release as re on re.artist_credit = ac.id
left join release_status as re_s on re_s.id = re.status
left join release_country as re_c on re_c.release = re.id
left join area as a on a.id = re_c.country 
limit 100;

*/


/*select * from cover_art_archive.cover_art limit 1;*/